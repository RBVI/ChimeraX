# -----------------------------------------------------------------------------
# Compute the area of one surface within a specified distance of another
# surface.
#
def contact_area(p1, p2, d, color = None, offset = None, slab = None,
                 smooth = False, optimize = True):

  v1, t1 = p1.geometry
  n1 = p1.normals
  v2, t2 = p2.geometry

  xf1, xf2 = p1.model.openState.xform, p2.model.openState.xform
  if xf2 != xf1:
    xf = xf1.inverse()
    xf.multiply(xf2)
    import Matrix
    Matrix.xform_points(v2, xf)

  dist = surface_distance(v1, v2, t2, d, optimize)
  
  v, n, t = patch_geometry(v1, n1, t1, dist, d)
  if len(t) == 0:
    return 0

  import MeasureVolume as mv
  area = mv.surface_area(v, t)

  if not color is None:
    if smooth:
      from _surface import smooth_vertex_positions
      sfactor, siter = 0.3, 2
      smooth_vertex_positions(v, t, sfactor, siter)
    if slab:
      create_patch(v, n, t, p1.model, color, slab = slab)
    elif offset:
      create_patch(v, n, t, p1.model, color, offset = offset)
    else:
      set_patch_color(p1, dist, d, color)

  return area

# -----------------------------------------------------------------------------
#
def surface_distance(v1, v2, t2, d, optimize = True):

  from _surface import surface_distance as surf_dist
  if optimize:
    from numpy import empty, float32
    dist = empty((len(v1),), float32)
    dist[:] = 2*d
    # Use only vertices within 2*d contact range.
    import _closepoints as cp
    i1, i2 = cp.find_close_points(cp.BOXES_METHOD, v1, v2, 2*d)
    if len(i1) > 0 and len(i2) > 0:
      v1r = v1[i1]
      s2 = set(i2)
      t2r = [tri for tri in t2 if tri[0] in s2 or tri[1] in s2 or tri[2] in s2]
      dr = surf_dist(v1r, v2, t2r)[:,0]
      dist[i1] = dr
  else:
    dist = surf_dist(v1, v2, t2)[:,0] # n by 5 array (d,x,y,z,side)
  return dist

# -----------------------------------------------------------------------------
#
def patch_geometry(vertices, normals, triangles, vdist, d):

  v = []
  n = []
  t = []
  vi = {}
  vadd = {}
  for tri in triangles:
    vc = [i for i in (0,1,2) if vdist[tri[i]] < d]
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
      v01 = add_vertex(v0, v1, vdist, d, vertices, vadd, v, normals, n)
      v02 = add_vertex(v0, v2, vdist, d, vertices, vadd, v, normals, n)
      t.append((vi[v0],v01,v02))
    elif len(vc) == 2:          # Two contact vertices.
      i = 3 - sum(vc)
      v0,v1,v2 = tuple(tri[i:]) + tuple(tri[:i])
      for ve in (v1,v2):
        if not ve in vi:
          vi[ve] = len(v)
          v.append(vertices[ve])
          n.append(normals[ve])
      v01 = add_vertex(v0, v1, vdist, d, vertices, vadd, v, normals, n)
      v02 = add_vertex(v0, v2, vdist, d, vertices, vadd, v, normals, n)
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
    from Matrix import normalize_vector
    n.append(normalize_vector(f*normals[v0] + (1-f)*normals[v1]))
  return vadd[e]

# -----------------------------------------------------------------------------
#
def set_patch_color(p, vdist, d, color):

  # Set color of contact area.
  vc = p.vertexColors
  if vc is None:
    from numpy import empty, float32
    vc = empty((p.vertexCount,4),float32)
    vc[:,:] = p.color
  for v,dv in enumerate(vdist):
    if dv <= d:
      vc[v,:] = color
  p.vertexColors = vc

# -----------------------------------------------------------------------------
#
def create_patch(v, n, t, surf, color, offset = 0, slab = None):

  from _surface import SurfaceModel
  s = SurfaceModel()
  s.name = 'contact patch'
  from chimera import openModels as om
  om.add([s])
  s.openState.xform = surf.openState.xform

  p = s.newPiece()
  p.color = color
  p.save_in_session = True

  if offset:
    vo = v.copy()
    vo += n*offset
    p.geometry = vo,t
    p.normals = n

  if slab:
    from Mask import depthmask
    vs, ns, ts = depthmask.slab_surface(v, t, n, slab, sharp_edges = True)
    p.geometry = vs,ts
    p.normals = ns

  return p
