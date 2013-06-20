# -----------------------------------------------------------------------------
# Points are in grid xyz coordinates.
#
def points_ijk_bounds(points, pad, data):

  xyz_min, xyz_max = bounding_box(points)

  box_min = map(lambda x, p=pad: x-p, xyz_min)
  box_max = map(lambda x, p=pad: x+p, xyz_max)
  corners = box_corners(box_min, box_max)

  ijk_min, ijk_max = bounding_box(map(data.xyz_to_ijk, corners))

  # TODO: This is not the tightest bounding box for skewed boxes.
  #     Should take ijk min/max then add size of padding skewed to ijk space.

  return ijk_min, ijk_max

# -----------------------------------------------------------------------------
#
def bounding_box(points):

  xyz_min = [None, None, None]
  xyz_max = [None, None, None]
  for xyz in points:
    for a in range(3):
      if xyz_min[a] == None or xyz[a] < xyz_min[a]:
        xyz_min[a] = xyz[a]
      if xyz_max[a] == None or xyz[a] > xyz_max[a]:
        xyz_max[a] = xyz[a]
  return xyz_min, xyz_max

# -----------------------------------------------------------------------------
#
def clamp_region(region, size):

  ijk_min, ijk_max = region
  ijk_min = [max(v, 0) for v in ijk_min]
  r = (tuple([max(0, min(ijk_min[a], size[a]-1)) for a in (0,1,2)]),
       tuple([max(0, min(ijk_max[a], size[a]-1)) for a in (0,1,2)]))
  return r

# -----------------------------------------------------------------------------
#
def integer_region(region):

  ijk_min, ijk_max = region
  from math import floor, ceil
  r = (tuple([int(floor(i)) for i in ijk_min]),
       tuple([int(ceil(i)) for i in ijk_max]))
  return r

# -----------------------------------------------------------------------------
#
def box_corners(xyz_min, xyz_max):

  x0,y0,z0 = xyz_min
  x1,y1,z1 = xyz_max
  return ((x0,y0,z0), (x0,y0,z1), (x0,y1,z0), (x0,y1,z1),
          (x1,y0,z0), (x1,y0,z1), (x1,y1,z0), (x1,y1,z1))
