# Bounding box computations

def point_bounds(xyz, placements = []):

  if len(xyz) == 0:
    return None

  from numpy import array, ndarray
  axyz = xyz if isinstance(xyz, ndarray) else array(xyz)

  if placements:
    from numpy import empty, float32
    n = len(placements)
    xyz0 = empty((n,3), float32)
    xyz1 = empty((n,3), float32)
    txyz = empty(axyz.shape, float32)
    for i, tf in enumerate(transforms):
      txyz[:] = axyz
      tf.move(txyz)
      xyz0[i,:], xyz1[i,:] = txyz.min(axis=0), txyz.max(axis=0)
    xyz_min, xyz_max = xyz0.min(axis = 0), xyz1.max(axis = 0)
  else:
    xyz_min, xyz_max = xyz.min(axis=0), xyz.max(axis=0)

  return xyz_min, xyz_max

def union_bounds(blist):
  xyz_min, xyz_max = None, None
  for b in blist:
    if b is None or b == (None, None):
      continue
    pmin, pmax = b
    if xyz_min is None:
      xyz_min, xyz_max = pmin, pmax
    else:
      xyz_min = tuple(min(x,px) for x,px in zip(xyz_min, pmin))
      xyz_max = tuple(max(x,px) for x,px in zip(xyz_max, pmax))
  return None if xyz_min is None else (xyz_min, xyz_max)

def copies_bounding_box(bounds, plist):
  (x0,y0,z0),(x1,y1,z1) = bounds
  corners = ((x0,y0,z0),(x1,y0,z0),(x0,y1,z0),(x1,y1,z0),
             (x0,y0,z1),(x1,y0,z1),(x0,y1,z1),(x1,y1,z1))
  b = union_bounds(point_bounds(p * corners) for p in plist)
  return b

def point_axis_bounds(points, axis):

  from numpy import dot
  pa = dot(points, axis)
  a2 = dot(axis, axis)
  return pa.min()/a2, pa.max()/a2

def bounds_center_and_radius(bounds):
  if bounds is None or bounds == (None, None):
    return None, None
  (xmin,ymin,zmin), (xmax,ymax,zmax) = bounds
  w = max(xmax-xmin, ymax-ymin, zmax-zmin)
  cx,cy,cz = 0.5*(xmin+xmax),0.5*(ymin+ymax),0.5*(zmin+zmax)
  from numpy import array
  return array((cx,cy,cz)), w
