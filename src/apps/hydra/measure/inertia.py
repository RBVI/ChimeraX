# -----------------------------------------------------------------------------
# Compute inertia tensor principle axes for surface based on area.
# Calculation weights vertices by 1/3 area of adjoining triangles.
#
def surface_inertia(drawings):

  vw = []
  from ..map.map_cpp import vertex_areas
  for d in drawings:
    for va, ta, positions in d.all_geometries():
      weights = vertex_areas(va, ta)
      for p in positions.place_list():
        v = p.moved(va)
        vw.append((v, weights))
    
  return moments_of_inertia(vw)

# -----------------------------------------------------------------------------
# Compute inertia tensor principle axes for atoms using atomic mass weights.
# Results are in eye coordinates.
#
def atoms_inertia(atoms):

  xyz = atoms.coordinates()
  from ..molecule.mass import element_mass
  weights = element_mass[atoms.element_numbers()]
  return moments_of_inertia([(xyz,weights)])

# -----------------------------------------------------------------------------
# Compute inertia tensor principle axes for mass above lowest contour level
# of density map.
#
def map_inertia(maps):

  vw = [map_points_and_weights(v) for v in maps]
  return moments_of_inertia(vw)

# -----------------------------------------------------------------------------
#
def map_points_and_weights(v, level = None, step = None, subregion = None):

  if level is None:
    if len(v.surface_levels) == 0:
      from numpy import empty, float32
      return empty((0,3),float32), empty((0,),float32)
    # Use lowest displayed contour level.
    level = min(v.surface_levels)

  # Get 3-d array of map values.
  m = v.matrix(step = step, subregion = subregion)

  from ..map import map_cpp
  points_int = map_cpp.high_indices(m, level)
  from numpy import float32
  points = points_int.astype(float32)
  tf = v.matrix_indices_to_xyz_transform(step, subregion)
  tf.move(points)
  weights = m[points_int[:,2],points_int[:,1],points_int[:,0]]

  return points, weights

# -----------------------------------------------------------------------------
# Compute inertia axes and moments for weighted set of points.
# Takes list of paired vertex and weight arrays.
#
def moments_of_inertia(vw):

  from numpy import zeros, float, array, dot, outer, argsort, linalg, identity
  i = zeros((3,3), float)
  c = zeros((3,), float)
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
  from ..geometry.vector import inner_product, cross_product
  if inner_product(cross_product(axes[0],axes[1]),axes[2]) < 0:
    axes[2,:] = -axes[2,:]  # Make axes a right handed coordinate system

  # Make rows of 3 by 3 matrix the principle axes.
  return axes, seval, c

# -----------------------------------------------------------------------------
#
def ellipsoid_surface(axes, lengths, center, color, surface):

  xf = surface.position.inverse()
  sa, sc = transform_ellipsoid(axes, center, xf)
  varray, narray, tarray = ellipsoid_geometry(sc, sa, lengths)
  d = surface.new_drawing()
  d.geometry = varray, tarray
  d.normals = narray
  d.color = color
  return d

# -----------------------------------------------------------------------------
#
def ellipsoid_geometry(center, axes, axis_lengths):

  from ..surface import shapes
  varray, narray, tarray = shapes.sphere_geometry(1280)
  narray = narray.copy()        # Is same as varray for sphere.
  from ..geometry import place, vector
  ptf = place.Place(axes = axes, origin = center) * place.scale(axis_lengths)
  ptf.move(varray)
  ntf = place.Place(axes = axes) * place.scale([1/l for l in axis_lengths])
  ntf.move(narray)
  vector.normalize_vectors(narray)

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
  from ..map.map_cpp import vertex_areas
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
    axes = place.apply_without_translation(axes)
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

  tf = surfs[0].position        # Axes reported relative to first surface
  info = axes_info(axes, d2, elen, center, tf.inverse())

  if surface:
    if color is None:
      from numpy import mean, uint8
      color = mean([s.color for s in surfs], axis = 0).astype(uint8)
    ellipsoid_surface(axes, elen, center, color, surface)

  return info

# -----------------------------------------------------------------------------
#
def atoms_inertia_ellipsoid(atoms = True, color = None, surface = None):

  if len(atoms) == 0:
    return

  axes, d2, center = atoms_inertia(atoms)
  elen = inertia_ellipsoid_size(d2)

  m0 = atoms.molecules()[0]
  pl = m0.position
  info = axes_info(axes, d2, elen, center, pl.inverse())

  if surface:
    if color is None:
      from numpy import uint8
      color = atoms.colors().mean(axis=0).astype(uint8) # Average color of atoms
    ellipsoid_surface(axes, elen, center, color, surface)

  return info

# -----------------------------------------------------------------------------
#
def transform_ellipsoid(axes, center, tf):

  axes = tf.apply_without_translation(axes)
  center = tf * center
  return axes, center

# -----------------------------------------------------------------------------
# Inertia of mass of density map.  Uniform solid ellipsoid with matching inertia shown.
#
def density_map_inertia_ellipsoid(maps, color = None, surface = None):

  if len(maps) == 0:
    return None

  axes, d2, center = map_inertia(maps)
  if axes is None:
    return None
  elen = inertia_ellipsoid_size(d2)

  tf = maps[0].position        # Axes reported relative to first map
  info = axes_info(axes, d2, elen, center, tf.inverse())

  if surface:
    if color is None:
      from numpy import mean, uint8
      color = (255*mean([m.surface_colors[0] for m in maps], axis = 0)).astype(uint8)
    ellipsoid_surface(axes, elen, center, color, surface)

  return info
