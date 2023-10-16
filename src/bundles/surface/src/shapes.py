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

  return va, va.copy(), ta


# -----------------------------------------------------------------------------
#
def sphere_geometry2(ntri):
    '''
    Return vertex, normal vector and triangle arrays for unit sphere geometry.
    Alternate techinque that produces any even number of triangles >= 4.
    Use in place of :py:func:`sphere_geometry` in new code.
    '''
    from chimerax.geometry import sphere
    va, ta = sphere.sphere_triangulation(ntri)
    return va, va.copy(), ta


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
    #
    # NOTE: resize does not zero out the array on resize! It's fine
    # here, we fill in the array. But we must make sure not to allow
    # trash values through in future refactors.
    from numpy import resize
    vc = varray.shape[0]
    varray = resize(varray, (vc+2*nc+2,3))
    narray = resize(narray, (vc+2*nc+2,3))
    varray[vc:vc+nc,:] = varray[0:nc,:] # Copy circle
    varray[vc+nc,:] = (0,0,-0.5*height) # Center of circle
    varray[vc+nc+1:vc+2*nc+1,:] = varray[vc-nc:vc,:] # Copy circle
    varray[vc+2*nc+1,:] = (0,0,0.5*height) # Center of circle
    narray[vc:vc+nc+1,:] = (0,0,-1)
    narray[vc+nc+1:,:] = (0,0,1)

    tc = tarray.shape[0]
    tarray = resize(tarray, (tc+2*nc,3))
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
    if segments == 0:
      return va, na, ta
    h = 0.5/segments
    va[:,2] *= h
    nv = len(va)
    vs = []
    ns = []
    ts = []
    for s in range(segments):
        v = va.copy()
        v[:,2] += (s - (segments-1)/2)*2*h*height
        vs.append(v)
        ns.append(na)
        ts.append(ta + s*nv)
    from numpy import concatenate
    vd, nd, td = concatenate(vs), concatenate(ns), concatenate(ts)
    return vd, nd, td


# -----------------------------------------------------------------------------
#
def cone_geometry(radius = 1, height = 1, nc = 20, caps = True, points_up = True):
    '''
    Return vertex, normal vector and triangle arrays for cone geometry
    with specified radius and height with middle of cone at origin
    '''
    from numpy import ones, empty, float32, arange, cos, sin, int32, pi
    vc = nc * 2
    tc = nc
    if caps:
        vc += (nc + 1)
        tc += nc
    varray = empty((vc, 3), float32)
    narray = empty((vc, 3), float32)
    tarray = empty((tc, 3), int32)

    # Compute a circle (which may be used twice if caps is true)
    angles = (2 * pi / nc) * arange(nc)
    import sys
    circle = empty((nc, 2), float32)
    circle[:,0] = cos(angles)
    circle[:,1] = sin(angles) if points_up else -sin(angles)


    # Create cone faces (first nc*2 vertices)
    zbase = -0.5*height if points_up else 0.5*height
    znorm = radius/height if points_up else -radius/height
    nc2 = nc * 2
    varray[:nc] = (0, 0, -zbase)      # point of cone (multiple normals)
    narray[:nc,:2] = circle
    narray[:nc,2] = znorm
    varray[nc:nc2,:2] = radius*circle      # base of cone
    varray[nc:nc2,2] = zbase
    narray[nc:nc2,:2] = circle
    narray[nc:nc2,2] = znorm
    tarray[:nc,0] = arange(nc)
    tarray[:nc,1] = arange(nc) + nc
    tarray[:nc,2] = (arange(nc) + 1) % nc + nc
    from chimerax.geometry import normalize_vectors
    normalize_vectors(narray[:nc2])
    
    # Create cone base (last nc+1 vertices)
    if caps:
        varray[nc2] = (0, 0, zbase)
        varray[nc2+1:,:2] = radius*circle
        varray[nc2+1:,2] = zbase
        narray[nc2:] = (0,0,-1) if points_up else (0,0,1)
        tarray[nc:,0] = nc2
        tarray[nc:,1] = (arange(nc) + 1) % nc + nc2 + 1
        tarray[nc:,2] = arange(nc) + nc2 + 1

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

def box_geometry(llb, urf):
    '''
    Return vertex, normal vector and triangle arrays for box with
    corners llb and urf (lower-left-back, upper-right-front)
    '''
    #       v2 ---- v3
    #        |\      |\
    #        | v6 ---- v7 = urf
    #        |  |    | |
    #        |  |    | |
    # llb = v0 -|---v1 |
    #         \ |     \|
    #          v4 ---- v5
    from numpy import array, float32, int32
    vertices = array([
        # -x, v0-v4-v2-v6
        [llb[0], llb[1], llb[2]],
        [llb[0], llb[1], urf[2]],
        [llb[0], urf[1], llb[2]],
        [llb[0], urf[1], urf[2]],

        # -y, v0-v1-v4-v5
        [llb[0], llb[1], llb[2]],
        [urf[0], llb[1], llb[2]],
        [llb[0], llb[1], urf[2]],
        [urf[0], llb[1], urf[2]],

        # -z, v1-v0-v3-v2
        [urf[0], llb[1], llb[2]],
        [llb[0], llb[1], llb[2]],
        [urf[0], urf[1], llb[2]],
        [llb[0], urf[1], llb[2]],

        # x, v5-v1-v7-v3
        [urf[0], llb[1], urf[2]],
        [urf[0], llb[1], llb[2]],
        [urf[0], urf[1], urf[2]],
        [urf[0], urf[1], llb[2]],

        # y, v3-v2-v7-v6
        [urf[0], urf[1], llb[2]],
        [llb[0], urf[1], llb[2]],
        [urf[0], urf[1], urf[2]],
        [llb[0], urf[1], urf[2]],

        # z, v4-v5-v6-v7
        [llb[0], llb[1], urf[2]],
        [urf[0], llb[1], urf[2]],
        [llb[0], urf[1], urf[2]],
        [urf[0], urf[1], urf[2]],
    ], dtype=float32)

    normals = array([
        # -x, v0-v4-v2-v6
        [-1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],

        # -y, v0-v1-v4-v5
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],

        # -z, v1-v0-v3-v2
        [0, 0, -1],
        [0, 0, -1],
        [0, 0, -1],
        [0, 0, -1],

        # x, v5-v1-v7-v3
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],

        # y, v3-v2-v7-v6
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],

        # z, v4-v5-v6-v7
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=float32)
    triangles = array([
        [0, 1, 2], [2, 1, 3],           # -x
        [4, 5, 6], [6, 5, 7],           # -y
        [8, 9, 10], [10, 9, 11],        # -z
        [12, 13, 14], [14, 13, 15],     # x
        [16, 17, 18], [18, 17, 19],     # y
        [20, 21, 22], [22, 21, 23],     # z
    ], dtype=int32)
    return vertices, normals, triangles
