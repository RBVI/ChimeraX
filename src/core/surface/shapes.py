# vim: set expandtab shiftwidth=4 softtabstop=4:

def sphere_geometry(ntri):
  '''
  Return vertex, normal vector and triangle arrays for unit sphere geometry.
  Only produces 20, 80, 320, ... (multiples of 4) triangle count.
  '''
  from . import icosahedron
  va, ta = icosahedron.icosahedron_geometry()
  from numpy import int32
  ta = ta.astype(int32)

  # Subdivide triangles
  while 4*len(ta) <= ntri:
    from ._surface import subdivide_triangles
    va, ta = subdivide_triangles(va, ta)

  # Put all vertices on sphere.
  from numpy import sqrt
  vn = sqrt((va*va).sum(axis = 1))
  for a in (0,1,2):
    va[:,a] /= vn

  return va, va, ta


# -----------------------------------------------------------------------------
#
def cylinder_geometry(radius = 1, height = 1, nz = 2, nc = 10, caps = True,
                      hexagonal_lattice = False):
    '''
    Return vertex, normal vector and triangle arrays for cylinder geometry
    with specified radius and height centered at the origin.
    '''
    varray, narray, tarray = unit_cylinder_geometry(nz, nc, hexagonal_lattice)
    varray[:,0] *= radius
    varray[:,1] *= radius
    varray[:,2] *= height
   
    if not caps:
        return varray, narray, tarray

    # Duplicate end rings to make sharp crease at cap edge.
    vc = varray.shape[0]
    varray.resize((vc+2*nc+2,3))
    narray.resize((vc+2*nc+2,3))
    varray[vc:vc+nc,:] = varray[0:nc,:] # Copy circle
    varray[vc+nc,:] = (0,0,-0.5*height) # Center of circle
    varray[vc+nc+1:vc+2*nc+1,:] = varray[vc-nc:vc,:] # Copy circle
    varray[vc+2*nc+1,:] = (0,0,0.5*height) # Center of circle
    narray[vc:vc+nc+1,:] = (0,0,-1)
    narray[vc+nc+1:,:] = (0,0,1)

    tc = tarray.shape[0]
    tarray.resize((tc+2*nc,3))
    for i in range(nc):
        tarray[tc+i,:] = (vc+nc,vc+(i+1)%nc,vc+i)
        tarray[tc+nc+i,:] = (vc+2*nc+1,vc+nc+1+i,vc+nc+1+(i+1)%nc)

    return varray, narray, tarray

# -----------------------------------------------------------------------------
# Build a hexagonal lattice tube
#
def unit_cylinder_geometry(nz, nc, hexagonal_lattice = False):

    from numpy import empty, float32, arange, cos, sin, int32, pi
    vc = nz*nc
    tc = (nz-1)*nc*2
    varray = empty((vc,3), float32)
    narray = empty((vc,3), float32)
    tarray = empty((tc,3), int32)

    # Calculate vertices
    v = varray.reshape((nz,nc,3))
    angles = (2*pi/nc)*arange(nc)
    if hexagonal_lattice:
      v[::2,:,0] = cos(angles)
      v[::2,:,1] = sin(angles)
      angles += pi/nc
      v[1::2,:,0] = cos(angles)
      v[1::2,:,1] = sin(angles)
    else:
      # Rectangular lattice
      v[:,:,0] = cos(angles)
      v[:,:,1] = sin(angles)
    for z in range(nz):
        v[z,:,2] = float(z)/(nz-1) - 0.5

    # Set normals
    narray[:,:] = varray
    narray[:,2] = 0

    # Create triangles
    t = tarray.reshape((nz-1,nc,6))
    c = arange(nc)
    c1 = (c+1)%nc
    t[:,:,0] = t[1::2,:,3] = c
    t[::2,:,1] = t[::2,:,3] = t[1::2,:,1] = c1
    t[::2,:,4] = t[1::2,:,2] = t[1::2,:,4] = c1+nc
    t[::2,:,2] = t[:,:,5] = c+nc
    for z in range(1,nz-1):
        t[z,:,:] += z*nc

    return varray, narray, tarray

def dashed_cylinder_geometry(segments = 5, radius = 1, height = 1, nz = 2, nc = 10, caps = True):
    '''
    Return vertex, normal vector and triangle arrays for a sequence of colinear cylinders.
    '''
    va, na, ta = cylinder_geometry(radius, height, nz, nc, caps)
    h = 0.5/segments
    va[:,2] *= h
    nv = len(va)
    vs = []
    ns = []
    ts = []
    for s in range(segments):
        v = va.copy()
        v[:,2] += (s - (segments-1)/2)*2*h
        vs.append(v)
        ns.append(na)
        ts.append(ta + s*nv)
    from numpy import concatenate
    vd, nd, td = concatenate(vs), concatenate(ns), concatenate(ts)
    return vd, nd, td


# -----------------------------------------------------------------------------
#
def cone_geometry(radius = 1, height = 1, nc = 10, caps = True):
    '''
    Return vertex, normal vector and triangle arrays for cone geometry
    with specified radius and height with point of cone at origin
    '''
    from numpy import ones, empty, float32, arange, cos, sin, int32, pi
    vc = nc * 2
    tc = nc
    if caps:
        vc += (nc + 1)
        tc += nc
    varray = empty((vc, 3), float32)
    narray = empty((vc, 3), float32)
    tarray = empty((tc, 3), float32)

    # Compute a circle (which may be used twice if caps is true)
    angles = (2 * pi / nc) * arange(nc)
    import sys
    circle = empty((nc, 2), float32)
    circle[:,0] = cos(angles)
    circle[:,1] = sin(angles)

    # Create cone faces (first nc*2 vertices)
    # The normals are wrong, but let's see how they look
    nc2 = nc * 2
    varray[:nc] = (0, 0, -0.5)      # point of cone (multiple normals)
    narray[:nc,:2] = circle
    narray[:nc,2] = 0
    varray[nc:nc2,:2] = circle      # base of cone
    varray[nc:nc2,2] = 0.5
    narray[nc:nc2,:2] = circle      # wrong, but close (enough?)
    narray[nc:nc2,2] = 0
    tarray[:nc,0] = arange(nc)
    tarray[:nc,1] = (arange(nc) + 1) % nc + nc
    tarray[:nc,2] = arange(nc) + nc

    # Create cone base (last nc+1 vertices)
    if caps:
        varray[nc2] = (0, 0, 0.5)
        varray[nc2+1:,:2] = circle
        varray[nc2+1:,2] = 0.5
        narray[nc2:] = (0, 0, 1)
        tarray[nc2:,0] = nc2
        tarray[nc2:,1] = (arange(nc) + 1) % nc + nc2 + 1
        tarray[nc2:,2] = arange(nc) + nc2 + 1

    return varray, narray, tarray

def octahedron_geometry():
  '''
  Return vertex, normal vector and triangle arrays for a radius 1 octahedron.
  '''
  from numpy import array, float32, int32
  va = array(((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)), float32)
  ta = array(((0,2,4),(2,1,4),(1,3,4),(3,2,4),
              (2,0,5),(1,2,5),(3,1,5),(2,3,5)), int32)
  return va, va, ta

def tetrahedron_geometry():
  '''
  Return vertex, normal vector and triangle arrays for a radius 1 tetrahedron.
  '''
  from numpy import array, float32, int32, sqrt
  s = 1.0/sqrt(3)
  va = array(((s,s,s),(s,-s,-s),(-s,s,-s),(-s,-s,s)), float32)
  ta = array(((0,1,2),(0,2,3),(0,3,1),(2,1,3)), int32)
  return va, va, ta
