# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Compute the area of one surface within a specified distance of another
# surface.
#
def measure_contact_area(session, surface, with_surface = None, distance = 3, show = True,
                         color = (255,0,0,255), offset = 1.0, slab = None, smooth = False, optimize = True):

  v1, n1, t1 = surface.vertices, surface.normals, surface.triangles
  v2, t2 = with_surface.vertices, with_surface.triangles

  pos1, pos2 = surface.scene_position, with_surface.scene_position
  if pos2 != pos1:
    pos2to1 = pos1.inverse() * pos2
    pos2to1.transform_points(v2, in_place = True)

  dist = surface_distance(v1, v2, t2, distance, optimize)
  
  v, n, t = patch_geometry(v1, n1, t1, dist, distance)
  if len(t) == 0:
    return 0

  from . import surface_area
  area = surface_area(v, t)

  if show:
    if smooth:
      from . import smooth_vertex_positions
      sfactor, siter = 0.3, 2
      smooth_vertex_positions(v, t, sfactor, siter)
    if slab is not None:
      create_patch(session, v, n, t, pos1, color, slab = slab)
    elif offset != 0:
      create_patch(session, v, n, t, pos1, color, offset = offset)
    else:
      set_patch_color(surface, dist, distance, color)

  return area

# -----------------------------------------------------------------------------
#
def surface_distance(v1, v2, t2, distance, optimize = True):

  from . import surface_distance
  if optimize:
    from numpy import empty, float32
    dist = empty((len(v1),), float32)
    dist[:] = 2*distance
    # Use only vertices within 2*d contact range.
    from chimerax.geometry import find_close_points
    i1, i2 = find_close_points(v1, v2, 2*distance)
    if len(i1) > 0 and len(i2) > 0:
      v1r = v1[i1]
      s2 = set(i2)
      t2r = [tri for tri in t2 if tri[0] in s2 or tri[1] in s2 or tri[2] in s2]
      dr = surface_distance(v1r, v2, t2r)[:,0]
      dist[i1] = dr
  else:
    dist = surface_distance(v1, v2, t2)[:,0] # n by 5 array (d,x,y,z,side)
  return dist

# -----------------------------------------------------------------------------
#
def patch_geometry(vertices, normals, triangles, vdist, distance):

  v = []
  n = []
  t = []
  vi = {}
  vadd = {}
  for tri in triangles:
    vc = [i for i in (0,1,2) if vdist[tri[i]] < distance]
    if len(vc) == 3:            # Three contact vertices.
      for ve in tri:
        if not ve in vi:
          vi[ve] = len(v)
          v.append(vertices[ve])
          n.append(normals[ve])
      t.append(tuple([vi[ve] for ve in tri]))
    elif len(vc) == 1:          # One contact vertex.
      v0,v1,v2 = tuple(tri[vc[0]:]) + tuple(tri[:vc[0]])
      if not v0 in vi:
        vi[v0] = len(v)
        v.append(vertices[v0])
        n.append(normals[v0])
      v01 = add_vertex(v0, v1, vdist, distance, vertices, vadd, v, normals, n)
      v02 = add_vertex(v0, v2, vdist, distance, vertices, vadd, v, normals, n)
      t.append((vi[v0],v01,v02))
    elif len(vc) == 2:          # Two contact vertices.
      i = 3 - sum(vc)
      v0,v1,v2 = tuple(tri[i:]) + tuple(tri[:i])
      for ve in (v1,v2):
        if not ve in vi:
          vi[ve] = len(v)
          v.append(vertices[ve])
          n.append(normals[ve])
      v01 = add_vertex(v0, v1, vdist, distance, vertices, vadd, v, normals, n)
      v02 = add_vertex(v0, v2, vdist, distance, vertices, vadd, v, normals, n)
      t.append((v01,vi[v1],vi[v2]))
      t.append((v01,vi[v2],v02))

  from numpy import array, float32, int32
  return array(v,float32), array(n,float32), array(t,int32)

# -----------------------------------------------------------------------------
#
def add_vertex(v0, v1, vdist, d, vertices, vadd, v, normals, n):

  e = (v0,v1) if v0 < v1 else (v1,v0)
  if not e in vadd:
    vadd[e] = len(v)
    f = (d - vdist[v1]) / (vdist[v0] - vdist[v1])
    v.append(f*vertices[v0] + (1-f)*vertices[v1])
    from chimerax.geometry import normalize_vector
    n.append(normalize_vector(f*normals[v0] + (1-f)*normals[v1]))
  return vadd[e]

# -----------------------------------------------------------------------------
#
def set_patch_color(surface, vdist, distance, color):

  # Set color of contact area.
  vc = surface.get_vertex_colors(create = True)
  for v,dv in enumerate(vdist):
    if dv <= distance:
      vc[v,:] = color
  surface.vertex_colors = vc

# -----------------------------------------------------------------------------
#
def create_patch(session, v, n, t, position, color, offset = 0, slab = None):

  from chimerax.core.models import Surface
  s = Surface('contact patch', session)
  s.SESSION_SAVE_DRAWING = True	# Save geometry in sessions
  s.position = position
  s.color = color

  if offset:
    vo = v.copy()
    vo += n*offset
    s.set_geometry(vo,n,t)

  if slab:
    if isinstance(slab, float):
      slab = (-0.5*slab, 0.5*slab)
    from chimerax.mask import depthmask
    vs, ns, ts = depthmask.slab_surface(v, t, n, slab, sharp_edges = True)
    s.set_geometry(vs,ns,ts)

  session.models.add([s])

  return s

# -----------------------------------------------------------------------------
#
def register_contactarea_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import SurfaceArg, FloatArg, Float2Arg, Color8Arg, BoolArg, Or
    desc = CmdDesc(
        required = [('surface', SurfaceArg)],
        keyword = [('with_surface', SurfaceArg),
                   ('distance', FloatArg),
                   ('show', BoolArg),
                   ('color', Color8Arg),
                   ('offset', FloatArg),
                   ('slab', Or(FloatArg, Float2Arg)),
                   ('smooth', BoolArg),
                   ('optimize', BoolArg),
                   ],
        required_arguments = ['with_surface'],
        synopsis = 'Compute and show contact are between two surfaces')
    register('measure contactArea', desc, measure_contact_area, logger=logger)
