# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def measure_inertia(session, objects, show_ellipsoid = True, color = None, per_chain = False,
                    model_id = None, replace = True):

    log = session.logger
    if color is not None:
        color = color.uint8x4()
    atoms = objects.atoms
    if atoms:
        mols = atoms.unique_structures
        mname = _molecules_name(mols)
        sname = ('ellipsoids ' if per_chain else 'ellipsoid ') + mname
        surf = _surface_model(sname, mols[0].scene_position, model_id, replace, session) if show_ellipsoid else None
        if per_chain:
            catoms = atoms.by_chain
            for mol, cid, cat in catoms:
                info = atoms_inertia_ellipsoid(cat, color, surf, submodel_name = cid)
                log.info('Inertia axes for %s, chain %s, %d atoms\n%s'
                         % (mname, cid, len(cat), info))
        else:
            info = atoms_inertia_ellipsoid(atoms, color, surf)
            log.info('Inertia axes for %s, %d atoms\n%s' % (mname, len(atoms), info))

    from chimerax.core.models import Surface
    from chimerax.map.volume import VolumeSurface
    surfs = [s for s in objects.models if isinstance(s, Surface) and not isinstance(s, VolumeSurface)]
    if surfs:
        sname = 'ellipsoid ' + (surfs[0].name if len(surfs) == 1 else ('%d surfaces' % len(surfs)))
        surf = _surface_model(sname, surfs[0].scene_position, model_id, replace, session) if show_ellipsoid else None
        info = surface_inertia_ellipsoid(surfs, color, surf)
        log.info('Inertia axes for %s\n%s' % (sname, info))

    from chimerax.map import Volume
    maps = [v for v in objects.models if isinstance(v, Volume)]
    if maps:
        mname = 'ellipsoid ' + (maps[0].name if len(maps) == 1 else ('%d maps' % len(maps)))
        surf = _surface_model(mname, maps[0].scene_position, model_id, replace, session) if show_ellipsoid else None
        info = density_map_inertia_ellipsoid(maps, color, surf)
        log.info('Inertia axes for %s\n%s' % (mname, info))

    if not (atoms or surfs or maps):
        log.info('No atoms, surfaces or volumes specified')

# -----------------------------------------------------------------------------
#
def _surface_model(name, place, model_id, replace, session):

    from chimerax.core.models import Surface
    if not model_id is None:
        slist = session.models.list(model_id = model_id, type = Surface)
        if slist:
            s = slist[0]
            if replace:
                session.models.close([s])
            else:
                return s

    s = Surface(name, session)
    s.id = model_id
    session.models.add([s])
    s.scene_position = place
    return s

# ----------------------------------------------------------------------------
#        
def _molecules_name(mlist):

    if len(mlist) == 1:
        return mlist[0].name
    return '%d molecules' % len(mlist)

# -----------------------------------------------------------------------------
# Compute inertia tensor principle axes for surface based on area.
# Calculation weights vertices by 1/3 area of adjoining triangles.
#
def surface_inertia(surfs):

  vw = []
  from chimerax.surface import vertex_areas
  for s in surfs:
      va, ta = s.vertices, s.triangles
      weights = vertex_areas(va, ta)
      v = s.scene_position.transform_points(va)
      vw.append((v, weights))
    
  return moments_of_inertia(vw)

# -----------------------------------------------------------------------------
# Compute inertia tensor principle axes for atoms using atomic mass weights.
# Results are in eye coordinates.
#
def atoms_inertia(atoms):

  xyz = atoms.scene_coords
  weights = atoms.elements.masses
  return moments_of_inertia([(xyz,weights)])

# -----------------------------------------------------------------------------
# Compute inertia tensor principle axes for mass above lowest contour level
# of density map.
#
def map_inertia(maps):

  vw = [map_points_and_weights(v, scene_coordinates = True) for v in maps]
  return moments_of_inertia(vw)

# -----------------------------------------------------------------------------
#
def map_points_and_weights(v, level = None, step = None, subregion = None,
                           scene_coordinates = False):

  if level is None:
    if len(v.surfaces) == 0:
      from numpy import empty, float32
      return empty((0,3),float32), empty((0,),float32)
    # Use lowest displayed contour level.
    level = v.minimum_surface_level

  # Get 3-d array of map values.
  m = v.matrix(step = step, subregion = subregion)

  from chimerax.map import high_indices
  points_int = high_indices(m, level)
  from numpy import float32
  points = points_int.astype(float32)
  tf = v.matrix_indices_to_xyz_transform(step, subregion)
  if scene_coordinates:
      tf = v.scene_position * tf
  tf.transform_points(points, in_place = True)
  weights = m[points_int[:,2],points_int[:,1],points_int[:,0]]

  return points, weights

# -----------------------------------------------------------------------------
# Compute inertia axes and moments for weighted set of points.
# Takes list of paired vertex and weight arrays.
#
def moments_of_inertia(vw):

  from numpy import zeros, float64, array, dot, outer, argsort, linalg, identity
  i = zeros((3,3), float64)
  c = zeros((3,), float64)
  w = 0
  for xyz, weights in vw:
    xyz, weights = array(xyz), array(weights)
    n = len(xyz)
    if n > 0 :
      wxyz = weights.reshape((n,1)) * xyz
      w += weights.sum()
      i += (xyz*wxyz).sum()*identity(3) - dot(xyz.transpose(),wxyz)
      c += wxyz.sum(axis = 0)

  if w == 0:
    return None, None, None      # All weights are zero.

  i /= w
  c /= w                         # Center of vertices
  i -= dot(c,c)*identity(3) - outer(c,c)

  eval, evect = linalg.eigh(i)

  # Sort by eigenvalue size.
  order = argsort(eval)
  seval = eval[order]
  sevect = evect[:,order]

  axes = sevect.transpose()
  from chimerax.geometry import inner_product, cross_product
  if inner_product(cross_product(axes[0],axes[1]),axes[2]) < 0:
    axes[2,:] = -axes[2,:]  # Make axes a right handed coordinate system

  # Make rows of 3 by 3 matrix the principle axes.
  return axes, seval, c

# -----------------------------------------------------------------------------
#
def ellipsoid_surface(axes, lengths, center, color, surface, submodel_name = None,
                      num_triangles = 1000):

  xf = surface.scene_position.inverse()
  sa, sc = transform_ellipsoid(axes, center, xf)
  varray, narray, tarray = ellipsoid_geometry(sc, sa, lengths, num_triangles = num_triangles)
  if submodel_name is None:
      s = surface
  else:
      from chimerax.core.models import Surface
      s = Surface(submodel_name, surface.session)
      surface.add([s])
  s.set_geometry(varray, narray, tarray)
  s.color = color
  return s

# -----------------------------------------------------------------------------
#
def ellipsoid_geometry(center, axes, axis_lengths, num_triangles = 1000):

  from chimerax.surface import sphere_geometry
  varray, narray, tarray = sphere_geometry(num_triangles)
  narray = narray.copy()        # Is same as varray for sphere.
  from chimerax.geometry import Place, scale, normalize_vectors
  ptf = Place(axes = axes, origin = center) * scale(axis_lengths)
  ptf.transform_points(varray, in_place = True)
  ntf = Place(axes = axes) * scale([1/l for l in axis_lengths])
  ntf.transform_vectors(narray, in_place = True)
  normalize_vectors(narray)

  return varray, narray, tarray

# -----------------------------------------------------------------------------
#
def inertia_ellipsoid_size(d2, shell = False):

  if shell:
    # Match inertia of uniform thickness ellipsoidal shell.
    elen = ellipsoid_shell_size_from_moments(d2)
  else:
    # Solid ellipsoid inertia about "a" axis = m*(b*b + c*c)/5
    d2sum = sum(d2)
    from math import sqrt
    elen = [sqrt(5*max(0,(0.5*d2sum - d2[a]))) for a in range(3)]
  return elen

# -----------------------------------------------------------------------------
# There is probably no simple formula for moments of inertia of a uniform
# thickness ellipsoid shell (likely elliptic integrals).
# A non-uniform thickness shell thicker along longer axes has moment of
# inertia I_a = m*(b*b + c*c)/3.
# This routines uses an iterative method to find ellipsoid axis lengths with
# specified moments for a uniform thickness shell.
#
# TODO: Convergence is poor for long aspect (10:1) ellipsoids.  With 10
#       iterations, sizes in small dimensions off by ~5%.
#
def ellipsoid_shell_size_from_moments(d2):

  d2sum = sum(d2)
  from math import sqrt
  elen = [sqrt(max(0,3*(0.5*d2sum - d2[a]))) for a in range(3)]
  varray, narray, tarray = ellipsoid_geometry(center = (0,0,0),
                                              axes = ((1,0,0),(0,1,0),(0,0,1)),
                                              axis_lengths = elen)
  from chimerax.surface import vertex_areas
  for k in range(10):
    weights = vertex_areas(varray, tarray)
    axes, d2e, center = moments_of_inertia([(varray, weights)])
    de = (d2 - d2e) / d2
    escale = 0.25*(-2*de+de.sum()) + 1
    for a in range(3):
      varray[:,a] *= escale[a]
    elen = [elen[a]*escale[a] for a in range(3)]
  return elen

# -----------------------------------------------------------------------------
#
def axes_info(axes, d2, elen, center, place = None):

  if place:
    axes = place.transform_vectors(axes)
    center = place * center
  from math import sqrt
  paxes = ['\tv%d = %6.3f %6.3f %6.3f   %s = %6.3f   r%d = %6.3f' %
           (a+1, axes[a][0], axes[a][1], axes[a][2],
            ('a','b','c')[a], elen[a], a+1, sqrt(d2[a]))
           for a in range(3)]
  c = '\tcenter = %8.5g %8.5g %8.5g' % tuple(center)
  info = '%s\n%s\n' % ('\n'.join(paxes), c)
  return info

# -----------------------------------------------------------------------------
# Inertia of uniform thickness surface.  Uniform ellipsoidal shell with
# matching inertia per area shown.
#
def surface_inertia_ellipsoid(surfs, color = None, surface = None):

  if len(surfs) == 0:
    return

  axes, d2, center = surface_inertia(surfs)
  elen = inertia_ellipsoid_size(d2, shell = True)

  tf = surfs[0].scene_position        # Axes reported relative to first surface
  info = axes_info(axes, d2, elen, center, tf.inverse())

  if surface:
    if color is None:
      from numpy import mean, uint8
      color = mean([s.color for s in surfs], axis = 0).astype(uint8)
    ellipsoid_surface(axes, elen, center, color, surface)

  return info

# -----------------------------------------------------------------------------
#
def atoms_inertia_ellipsoid(atoms, color = None, surface = None, submodel_name = None):

  if len(atoms) == 0:
    return

  axes, d2, center = atoms_inertia(atoms)
  elen = inertia_ellipsoid_size(d2)

  m0 = atoms[0].structure
  pl = m0.scene_position
  info = axes_info(axes, d2, elen, center, pl.inverse())

  if surface:
    if color is None:
      from numpy import uint8
      color = atoms.colors.mean(axis=0).astype(uint8) # Average color of atoms
    ellipsoid_surface(axes, elen, center, color, surface, submodel_name = submodel_name)

  return info

# -----------------------------------------------------------------------------
#
def transform_ellipsoid(axes, center, tf):

  axes = tf.transform_vectors(axes)
  center = tf * center
  return axes, center

# -----------------------------------------------------------------------------
# Inertia of mass of density map.  Uniform solid ellipsoid with matching inertia shown.
#
def density_map_inertia_ellipsoid(maps, color = None, surface = None):

  if len(maps) == 0:
    return None

  axes, d2, center = map_inertia(maps)	# Scene coordinates
  if axes is None:
    return None
  elen = inertia_ellipsoid_size(d2)

  tf = maps[0].scene_position        # Report axes relative to first map
  info = axes_info(axes, d2, elen, center, tf.inverse())

  if surface:
    if color is None:
      from numpy import mean, uint8
      color = mean([m.surfaces[0].color for m in maps], axis = 0).astype(uint8)
    ellipsoid_surface(axes, elen, center, color, surface)

  return info

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ObjectsArg, BoolArg, ColorArg, ModelIdArg
    desc = CmdDesc(
        required = [('objects', ObjectsArg)],
        keyword = [('show_ellipsoid', BoolArg),
                   ('color', ColorArg),
                   ('per_chain', BoolArg),
                   ('model_id', ModelIdArg),
                   ('replace', BoolArg),],
        synopsis = 'measure moments of inertia')
    register('measure inertia', desc, measure_inertia, logger=logger)
