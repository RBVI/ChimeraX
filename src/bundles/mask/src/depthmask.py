# -----------------------------------------------------------------------------
# Compute a 2-dimensional depth array from a list of triangles.  The
# vertex y and x coordinates are indices into the depth array and the
# z coordinate is the depth.  The depth array should be initialized to
# a desired maximum depth before being passed to this routine.  If a
# "beyond" array is passed it should be the same size as depth and
# only depths beyond its values will be recorded in the depth array.
# This can be used to get the second layer surface depth by passing in
# a "beyond" array that is the depth calculated for the first layer.
#
# Math needs to be done 64-bit to minimize round-off errors leading to
# multiple nearly identical depths at single grid points where there is only
# one surface point coincident with edge or vertex shared by multiple
# triangles.
#
def surface_z_depth(varray, triangles, depth, beyond = None):

  ysize, xsize = depth.shape

  # Record minimum depth for each triangle at array grid points.
  set = False

  from numpy import zeros, float64, array, float32
  tv = zeros((3,3), float64)    # Triangle vertices
  from math import floor, ceil
  for t in triangles:
    tv[0], tv[1], tv[2] = [varray[i] for i in t]
    ixmin, ixmax = tv[:,0].argmin(), tv[:,0].argmax()
    if ixmin == ixmax:
      continue      # Zero area triangle
    ixmid = 3 - (ixmin + ixmax)
    xmin, xmid, xmax = tv[ixmin,0], tv[ixmid,0], tv[ixmax,0]
    x0 = max(0,int(ceil(xmin)))
    x1 = min(xsize-1,int(floor(xmax)))+1
    for i in range(x0, x1):
      fxa = (i - xmin) / (xmax - xmin)
      ya,za = tv[ixmin,1:]*(1-fxa) + tv[ixmax,1:]*fxa
      if i < xmid:
        fxb = (i - xmin) / (xmid - xmin)
        yb,zb = tv[ixmin,1:]*(1-fxb) + tv[ixmid,1:]*fxb
      else:
        xsep = xmax - xmid
        if xsep == 0:
          fxb = 0
        else:
          fxb = (i - xmid) / xsep
        yb,zb = tv[ixmid,1:]*(1-fxb) + tv[ixmax,1:]*fxb
      if ya < yb:
        ymin,ymax,zmin,zmax = ya,yb,za,zb
      else:
        ymin,ymax,zmin,zmax = yb,ya,zb,za
      ysep = ymax - ymin
      y0 = max(0,int(ceil(ymin)))
      y1 = min(ysize-1,int(floor(ymax)))+1
      for j in range(y0, y1):
        if ysep == 0:
          fy = 0
        else:
          fy = (j - ymin) / ysep
        z = zmin*(1-fy) + zmax*fy
        if z < depth[j,i]:
          # Have to convert 64-bit z to 32-bit so same point does not appear
          # beyond itself.
          if beyond is None or array((z,),float32)[0] > beyond[j,i]:
            depth[j,i] = z
            set = True
  return set

# -----------------------------------------------------------------------------
#
def surfaces_z_depth(surfaces, depth, triangle_num,
                     beyond, beyond_triangle_num):

  set = False
  toffset = 0
  beyond_kw = {k:v for k,v in (('beyond', beyond), ('beyond_triangle_number', beyond_triangle_num)) if v is not None}
  from .mask_cpp import surface_z_depth
  for varray, tarray in surfaces:
    if surface_z_depth(varray, tarray, depth, triangle_num,
                       triangle_number_offset = toffset, **beyond_kw):
      set = True
    toffset += len(tarray)
  return set

# -----------------------------------------------------------------------------
# Vertices must be in volume local coordinates.
#
def masked_volume(volume, surfaces,
                  projection_axis = (0,0,1), full_map = False,
                  sandwich = False, invert_mask = False, fill_overlap = False,
                  extend = 0, model_id = None):

  # Calculate position of 2-d depth array and transform surfaces so projection
  # is along z axis.
  zsurf, size, tf = surface_projection_coordinates(surfaces, projection_axis,
                                                   volume)

  # Create minimal size volume mask array and calculate transformation from
  # mask indices to depth array indices.
  full = full_map or invert_mask
  vol, mvol, ijk_origin, mijk_to_dijk = volume_mask(volume, surfaces, full, tf)

  # Copy volume to masked volume at depth intervals inside surface.
  project_and_mask(zsurf, size, mvol, mijk_to_dijk, sandwich, fill_overlap)

  # Extend mask boundary by n voxels
  if extend:
    from .mask_cpp import pad_mask
    pad_mask(mvol, extend)

  # Multiply ones mask times volume.
  mvol *= vol

  # Invert mask
  if invert_mask:
    from numpy import subtract
    subtract(vol, mvol, mvol)

  # Create masked volume model.
  v = array_to_model(mvol, volume, ijk_origin, model_id)

  # Undisplay original map.
  volume.show(show = False)
  
  return v

# -----------------------------------------------------------------------------
# Calculate position of 2-d depth array and transform surfaces so projection
# is along z axis.
#
# If the projection axis is x, y or z make the projection grid exactly align
# with the volume grid but limited to the region covered by the surfaces.
#
def surface_projection_coordinates(surfaces, projection_axis, volume):

  g = volume.data

  # Scale rotated surface coordinates to grid index units.
  axis_aligned = (tuple(projection_axis) in ((1,0,0), (0,1,0), (0,0,1))
                  and tuple(g.cell_angles) == (90,90,90)
                  and g.rotation == ((1,0,0),(0,1,0),(0,0,1)))
  if axis_aligned:
    grid_spacing = g.step
  else:
    s = min(g.plane_spacings())
    grid_spacing = (s,s,s)

  # Determine transform from vertex coordinates to depth array indices
  # Rotate projection axis to z.
  from chimerax.geometry import orthonormal_frame, scale, translation
  tfrs = orthonormal_frame(projection_axis).inverse() * scale([1/s for s in grid_spacing])

  # Transform vertices to depth array coordinates.
  zsurf = []
  tcount = 0
  for vertices, triangles in surfaces:
    varray = tfrs.transform_points(vertices)
    zsurf.append((varray, triangles))
    tcount += len(triangles)
  if tcount == 0:
    return None

  # Compute origin for depth grid
  vmin, vmax = bounding_box(zsurf)
  if axis_aligned:
    o = tfrs * g.origin
    offset = [(vmin[a] - o[a]) for a in (0,1,2)]
    from math import floor
    align_frac = [offset[a] - floor(offset[a]) for a in (0,1,2)]
    vmin -= align_frac
  else:
    vmin -= 0.5

  tf = translation(-vmin) * tfrs

  # Shift surface vertices by depth grid origin
  for varray, triangles in zsurf:
    varray -= vmin

  # Compute size of depth grid
  from math import ceil
  size = tuple(int(ceil(vmax[a] - vmin[a] + 1)) for a in (0,1))

  return zsurf, size, tf

# -----------------------------------------------------------------------------
# Create minimal size volume mask array and calculate transformation from
# mask indices to depth array indices.
#
def volume_mask(volume, surfaces, full, tf):

  g = volume.data
  if full:
    from chimerax.map.volume import full_region
    ijk_min, ijk_max = full_region(g.size)[:2]
  else:
    ijk_min, ijk_max = bounding_box(surfaces, g.xyz_to_ijk_transform)
    from math import ceil, floor
    ijk_min = [int(floor(i)) for i in ijk_min]
    ijk_max = [int(ceil(i)) for i in ijk_max]
    from chimerax.map.volume import clamp_region
    ijk_min, ijk_max = clamp_region((ijk_min, ijk_max, (1,1,1)), g.size)[:2]
  ijk_size = [a-b+1 for a,b in zip(ijk_max, ijk_min)]
  vol = g.matrix(ijk_min, ijk_size)
  from numpy import zeros
  mvol = zeros(vol.shape, vol.dtype)
  from chimerax.geometry import translation
  mijk_to_dijk = tf * g.ijk_to_xyz_transform * translation(ijk_min)
  return vol, mvol, ijk_min, mijk_to_dijk

# -----------------------------------------------------------------------------
# Copy volume to masked volume at depth intervals inside surface.
#
def project_and_mask(zsurf, size, mvol, mijk_to_dijk, sandwich, fill_overlap):

  # Create projection depth arrays.
  from numpy import zeros, intc, float32
  shape = (size[1], size[0])
  depth = zeros(shape, float32)
  tnum = zeros(shape, intc)
  depth2 = zeros(shape, float32)
  tnum2 = zeros(shape, intc)

  # Copy volume to masked volume at masked depth intervals.
  max_depth = 1e37
  if sandwich:
    dlimit = .5*max_depth
  else:
    dlimit = 2*max_depth
  zsurfs = [[s] for s in zsurf] if fill_overlap else [zsurf]
  from .mask_cpp import fill_slab
  for zs in zsurfs:
    beyond = beyond_tnum = None
    max_layers = 200
    for iter in range(max_layers):
      depth.fill(max_depth)
      tnum.fill(-1)
      any = surfaces_z_depth(zs, depth, tnum, beyond, beyond_tnum)
      if not any:
        break
      depth2.fill(max_depth)
      tnum2.fill(-1)
      surfaces_z_depth(zs, depth2, tnum2, depth, tnum)
      fill_slab(depth, depth2, mijk_to_dijk.matrix, mvol, dlimit)
      beyond = depth2
      beyond_tnum = tnum2

# -----------------------------------------------------------------------------
# Create masked volume model from 3d array.
#
def array_to_model(mvol, volume, ijk_origin, model_id):

  # Create masked volume grid object.
  from chimerax.map_data import ArrayGridData
  g = volume.data
  morigin = g.ijk_to_xyz_transform * ijk_origin
  m = ArrayGridData(mvol, morigin, g.step, cell_angles = g.cell_angles,
                      rotation = g.rotation, name = g.name + ' masked')

  # Create masked volume object.
  from chimerax.map import volume_from_grid_data
  v = volume_from_grid_data(m, volume.session, model_id = model_id)
  v.copy_settings_from(volume, copy_region = False)
  v.show()

  return v

# -----------------------------------------------------------------------------
#
def copy_slab(depth, depth2, mijk_to_dijk, vol, mvol, dlimit):

  ksize, jsize, isize = mvol.shape
  djsize, disize = depth.shape
  for k in range(ksize):
    for j in range(jsize):
      for i in range(isize):
        di,dj,dk = mijk_to_dijk * (i,j,k)
        if di >= 0 and di < disize-1 and dj >= 0 and dj < djsize-1:
          # Interpolate depths, nearest neighbor
          # TODO: use linear interpolation.
          di = int(di + 0.5)
          dj = int(dj + 0.5)
          d1 = depth[dj,di]
          d2 = depth2[dj,di]
          if dk >= d1 and dk <= d2 and d1 <= dlimit and d2 <= dlimit:
            mvol[k,j,i] = vol[k,j,i]

# -----------------------------------------------------------------------------
#
def bounding_box(surfaces, tf = None):

  smin = smax = None
  from numpy import minimum, maximum
  for vertices, triangles in surfaces:
    if len(triangles) == 0:
      continue
    if tf is None:
      v = vertices
    else:
      v = tf.transform_points(vertices)
    v = v.take(triangles.ravel(), axis = 0)
    vmin = v.min(axis = 0)
    if smin is None:      smin = vmin
    else:                 smin = minimum(smin, vmin)
    vmax = v.max(axis = 0)
    if smax is None:      smax = vmax
    else:                 smax = maximum(smax, vmax)
  return smin, smax

# -----------------------------------------------------------------------------
# Pad can be one or two values.  If two values then a slab is formed by
# stitching offset copies of the surface at the boundary.
#
def surface_geometry(plist, tf, pad):

  surfaces = []
  for p in plist:
    surfs = []
    va = p.vertices
    ta = p.masked_triangles
    if va is None or len(va) == 0 or ta is None or len(ta) == 0:
      continue
    na = p.normals
    if isinstance(pad, (float,int)) and pad != 0:
      varray, tarray = offset_surface(va, ta, na, pad)
    elif isinstance(pad, (list,tuple)) and len(pad) == 2:
      varray, narray, tarray = slab_surface(va, ta, na, pad)
    else:
      varray, tarray = va, ta

    if not tf is None:
      vtf = tf * p.scene_position
      if not vtf.is_identity(tolerance = 0):
        varray = vtf.transform_points(varray)

    surfaces.append((varray, tarray))

  return surfaces

# -----------------------------------------------------------------------------
#
def offset_surface(varray, tarray, narray, pad):

  va = varray.copy()
  va += pad * narray
  return va, tarray

# -----------------------------------------------------------------------------
#
def slab_surface(va, ta, na, pad, sharp_edges = False):

  nv = len(va)
  nt = len(ta)

  from chimerax.surface import boundary_edges
  edges = boundary_edges(ta)
  ne = len(edges)
  if sharp_edges:
    nv2 = 4*nv
    nt2 = 2*nt+6*ne
  else:
    nv2 = 2*nv
    nt2 = 2*nt+2*ne
  from numpy import zeros
  varray = zeros((nv2,3), va.dtype)
  narray = zeros((nv2,3), na.dtype)
  tarray = zeros((nt2,3), ta.dtype)

  # Two copies of vertices offset along normals by padding values.
  varray[:nv,:] = va + pad[1] * na
  varray[nv:2*nv,:] = va + pad[0] * na
  narray[:nv,:] = na
  narray[nv:2*nv,:] = -na

  if sharp_edges:
    # TODO: Copy just edge vertices instead of all vertices when making
    #       sharp edges.
    e = edges[:,0]
    varray[2*nv:3*nv,:] = varray[:nv,:]
    varray[3*nv:,:] = varray[nv:2*nv,:]

  tarray[:nt,:] = ta
  # Reverse triangle orientation for inner face.
  tarray[nt:2*nt,0] = ta[:,0] + len(va)
  tarray[nt:2*nt,1] = ta[:,2] + len(va)
  tarray[nt:2*nt,2] = ta[:,1] + len(va)

  # Stitch faces with band of triangles, two per boundary edge.
  if sharp_edges:
    band_triangles(tarray[2*nt:,:], edges, 0, 2*nv)
    band_triangles(tarray[2*nt+2*ne:,:], edges, 2*nv, 3*nv)
    band_triangles(tarray[2*nt+4*ne:,:], edges, 3*nv, nv)
    # Set band normals.
    tband = tarray[2*nt+2*ne:2*nt+4*ne,:]
    from chimerax.surface import calculate_vertex_normals
    narray[2*nv:,:] = calculate_vertex_normals(varray, tband)[2*nv:,:]
  else:
    band_triangles(tarray[2*nt:,:], edges, 0, nv)

  return varray, narray, tarray

# -----------------------------------------------------------------------------
#
def band_triangles(tarray, edges, voffset0, voffset1):

  ne = len(edges)
  tarray[:ne,0] = edges[:,0] + voffset0
  tarray[:ne,1] = edges[:,0] + voffset1
  tarray[:ne,2] = edges[:,1] + voffset1
  tarray[ne:2*ne,0] = edges[:,0] + voffset0
  tarray[ne:2*ne,1] = edges[:,1] + voffset1
  tarray[ne:2*ne,2] = edges[:,1] + voffset0

