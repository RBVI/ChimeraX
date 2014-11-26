# Only produces 20, 80, 320, ... (multiples of 4) triangle count.
def sphere_geometry(ntri):
  from ..geometry import icosahedron
  va, ta = icosahedron.icosahedron_geometry()
  from numpy import int32, sqrt
  ta = ta.astype(int32)
  from ..map import map_cpp
  while 4*len(ta) <= ntri:
    va, ta = map_cpp.subdivide_triangles(va, ta)
  vn = sqrt((va*va).sum(axis = 1))
  for a in (0,1,2):
    va[:,a] /= vn
  return va, va, ta


# -----------------------------------------------------------------------------
#
def cylinder_geometry(radius = 1, height = 1, nz = 2, nc = 10, caps = True):

    varray, narray, tarray = unit_cylinder_geometry(nz, nc)
    varray[:,0] *= radius
    varray[:,1] *= radius
    varray[:,2] *= height
   
    if not caps:
        return varray, narray, tarray

    vc = varray.shape[0]
    varray.resize((vc+2,3))
    narray.resize((vc+2,3))
    varray[vc,:] = (0,0,-0.5*height)
    varray[vc+1,:] = (0,0,0.5*height)
    narray[vc,:] = (0,0,-1)
    narray[vc+1,:] = (0,0,1)

    tc = tarray.shape[0]
    tarray.resize((tc+2*nc,3))
    for i in range(nc):
        tarray[tc+i,:] = (vc,(i+1)%nc,i)
        tarray[tc+nc+i,:] = (vc+1,vc-nc+i,vc-nc+(i+1)%nc)

    return varray, narray, tarray

# -----------------------------------------------------------------------------
# Build a hexagonal lattice tube
#
def unit_cylinder_geometry(nz, nc):

    from numpy import empty, float32, arange, cos, sin, int32, pi
    vc = nz*nc
    tc = (nz-1)*nc*2
    varray = empty((vc,3), float32)
    narray = empty((vc,3), float32)
    tarray = empty((tc,3), int32)

    # Calculate vertices
    v = varray.reshape((nz,nc,3))
    angles = (2*pi/nc)*arange(nc)
    v[::2,:,0] = cos(angles)
    v[::2,:,1] = sin(angles)
    angles += pi/nc
    v[1::2,:,0] = cos(angles)
    v[1::2,:,1] = sin(angles)
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
