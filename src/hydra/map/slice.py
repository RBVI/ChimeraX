# -----------------------------------------------------------------------------
# Find points under the specified window position intersecting surface of
# a box.  The box is specified as (xyz_min, xyz_max).  The returned points
# are in box coordinates.  The segment inside the box is clipped by near
# and far clip planes.  It is also clipped by per-model clip planes if
# the clip_plane_model is provided.
#
def box_intercepts(line, box_to_line_transform, box, clip_plane_model = None):

  line_to_box_transform = box_to_line_transform.inverse()
  bline = line_to_box_transform * line
  xyz_in, xyz_out = box_line_intercepts(bline, box)
  if xyz_in == None or xyz_out == None:
    return xyz_in, xyz_out
  
  planes = per_model_clip_planes(clip_plane_model)
  xyz_in, xyz_out, f1, f2 = clip_segment_with_planes(xyz_in, xyz_out, planes)
  
  return xyz_in, xyz_out

# -----------------------------------------------------------------------------
# Returned line is transformed from screen coordinates using the given
# transform
#
def line_perpendicular_to_screen(line, line_to_result_transform):

  xyz_near, xyz_far = line_to_result_transform * line
  dir = tuple(a-b for a,b in zip(xyz_far, xyz_near))
  line = (xyz_near, dir)
  return line

# -----------------------------------------------------------------------------
# Returned planes are in screen coordinates.
# Normal points toward unclipped half-space.
#
def per_model_clip_planes(clip_plane_model):

  return []
  
  if clip_plane_model is None or not clip_plane_model.useClipPlane:
    return []
  
  p = clip_plane_model.clipPlane
  plane = (p.normal.data(), -p.offset())
  planes = [plane]

  # Handle slab clipping mode.
  if clip_plane_model.useClipThickness:
    normal = [-x for x in p.normal.data()]
    offset = p.offset() - clip_plane_model.clipThickness
    plane = (normal, offset)
    planes.append(plane)

  model_to_screen = clip_plane_model.position
  tplanes = [transform_plane(p, model_to_screen) for p in planes]

  return tplanes

# -----------------------------------------------------------------------------
# The transform need not be orthogonal.
#
def transform_plane(plane, transform):

  inv_tf = transform.inverse()
  inv_tf_transpose = inv_tf.transpose()
  normal, offset = plane
  n = inv_tf_transpose.apply_without_translation(normal)
  o = offset - (normal * (inv_tf * (0,0,0))).sum()
  return (n, o)

# -----------------------------------------------------------------------------
#
def box_line_intercepts(line, xyz_region):

  xyz_in = box_entry_point(line, xyz_region)
  xyz_out = box_entry_point(oppositely_directed_line(line), xyz_region)
  return xyz_in, xyz_out

# -----------------------------------------------------------------------------
#
def oppositely_directed_line(line):
  
  xyz1, xyz2 = line
  return (xyz2, xyz1)

# -----------------------------------------------------------------------------
# Place where directed line enters box, or None if no intersection.
#
def box_entry_point(line, xyz_region):

  xyz1, xyz2 = line
  p, d = xyz1, [a-b for a,b in zip(xyz2,xyz1)]
  xyz_min, xyz_max = xyz_region
  planes = (((1,0,0), xyz_min[0]), ((-1,0,0), -xyz_max[0]),
            ((0,1,0), xyz_min[1]), ((0,-1,0), -xyz_max[1]),
            ((0,0,1), xyz_min[2]), ((0,0,-1), -xyz_max[2]))
  for n, c in planes:
    nd = inner_product(n, d)
    if nd > 0:
      t = (c - inner_product(n,p)) / nd
      xyz = tuple(a + t*b for a,b in zip(p, d))
      outside = False
      for n2, c2 in planes:
        if n2 != n or c2 != c:
          if inner_product(xyz, n2) < c2:
            outside = True
            break
      if not outside:
        return xyz

  return None
  
# -----------------------------------------------------------------------------
# Plane normals point towards unclipped half-space.
#
def clip_segment_with_planes(xyz_1, xyz_2, planes):

  f1 = 0
  f2 = 1
  for normal, offset in planes:
    c1 = inner_product(normal, xyz_1) - offset
    c2 = inner_product(normal, xyz_2) - offset
    if c1 < 0 and c2 < 0:
      return None, None, None, None     # All of segment is clipped
    if c1 >= 0 and c2 >= 0:
      continue                          # None of segment is clipped
    f = c1 / (c1 - c2)
    if c1 < 0:
      f1 = max(f1, f)
    else:
      f2 = min(f2, f)

  if f1 == 0 and f2 == 1:
    return xyz_1, xyz_2, f1, f2

  if f1 > f2:
    return None, None, None, None

  i1 = [(1-f1)*a + f1*b for a,b in zip(xyz_1, xyz_2)]      # Intercept point
  i2 = [(1-f2)*a + f2*b for a,b in zip(xyz_1, xyz_2)]      # Intercept point
  return i1, i2, f1, f2

# -----------------------------------------------------------------------------
#
def inner_product(u, v):

  sum = 0
  for a in range(len(u)):
    sum = sum + u[a] * v[a]
  return sum

# -----------------------------------------------------------------------------
#
def nearest_volume_face(line, session):

    xyz1, xyz2 = line
    zmin = None
    hit = (None, None, None, None)
    from . import volume_list
    for v in volume_list(session):
        if v.shown():
          ijk, axis, side = face_intercept(v, line)
          if ijk:
            xyz = v.ijk_to_global_xyz(ijk)
            z = line_position(xyz, line)
            if zmin is None or z < zmin:
              hit = (v, axis, side, ijk)
              zmin = z
    return hit

# -----------------------------------------------------------------------------
#
def face_intercept(v, line):

  xyz1, xyz2 = line
  zmin = None
  hit = (None, None, None)
  if v.showing_orthoplanes():
    for ijk, axis in orthoplane_intercepts(v, line):
      xyz = v.ijk_to_global_xyz(ijk)
      z = line_position(xyz, line)
      if zmin is None or z < zmin:
        hit = (ijk, axis, 0)
        zmin = z
  else:
    ijk_in, ijk_out = volume_index_segment(v, line)
    if ijk_in:
      xyz = v.ijk_to_global_xyz(ijk_in)
      z = line_position(xyz, line)
      if zmin is None or z > zmin:
        axis, side = box_face_axis_side(ijk_in, v.region)
        hit = (ijk_in, axis, side)
        zmin = z
  return hit
  
# -----------------------------------------------------------------------------
#
def line_position(xyz, line):
  xyz1, xyz2 = line
  dxyz = xyz - xyz1
  xyz12 = xyz2 - xyz1
  from ..geometry import vector
  d2 = vector.norm(xyz12)
  f = vector.inner_product(dxyz,xyz12) / d2
  return f
  
# -----------------------------------------------------------------------------
#
def box_face_axis_side(ijk, region):

    dmin = None
    axis_side = None
    for axis in (0,1,2):
        for side in (0,1):
            d = abs(ijk[axis] - region[side][axis])
            if dmin is None or d < dmin:
                dmin = d
                axis_side = (axis, side)
    return axis_side

# -----------------------------------------------------------------------------
# Return intercept under mouse position with volume in volume xyz coordinates.
#
def volume_plane_intercept(line, volume, axis, i):

  ijk_to_scene = volume.model_transform() * volume.data.ijk_to_xyz_transform
  scene_to_ijk = ijk_to_scene.inverse()
  ijk1, ijk2 = scene_to_ijk * line
  p,d = ijk1, [a-b for a,b in zip(ijk2,ijk1)]
  if d[axis] == 0:
    return None
  t = (i - p[axis]) / d[axis]
  ijk = [p[a]+t*d[a] for a in (0,1,2)]
  ijk[axis] = i
  ijk_min, ijk_max = volume.region[:2]
  for a in (0,1,2):
    if ijk[a] < ijk_min[a] or ijk[a] > ijk_max[a]:
      return None
  return tuple(ijk)

# -----------------------------------------------------------------------------
#
def orthoplane_intercepts(v, line):

  ijk_axis = []
  for axis, p in v.shown_orthoplanes():
    ijk = volume_plane_intercept(line, v, axis, p)
    if not ijk is None:
      ijk_axis.append((ijk, axis))
  return ijk_axis

# -----------------------------------------------------------------------------
# Returns two intercept points with volume box in volume coordinates.
#
def volume_segment(volume, line):

  ijk_in, ijk_out = volume_index_segment(volume, line)
  d = volume.data
  xyz_in = None if ijk_in is None else d.ijk_to_xyz(ijk_in)
  xyz_out = None if ijk_out is None else d.ijk_to_xyz(ijk_out)

  return xyz_in, xyz_out

# -----------------------------------------------------------------------------
# line is in scene coordinates.
#
def volume_index_segment(volume, line, clipping_model = None):

  # box is in volume index coordinates, line in scene coordinates.
  box_to_line_transform = volume.position * volume.data.ijk_to_xyz_transform

  if clipping_model is None:
    clipping_model = volume

  ijk_in, ijk_out = box_intercepts(line, box_to_line_transform,
                                   volume.ijk_bounds(), clipping_model)
  
  return ijk_in, ijk_out

# -----------------------------------------------------------------------------
#
def slice_data_values(v, xyz_in, xyz_out):

  from ..geometry.vector import distance
  d = distance(xyz_in, xyz_out)
  #
  # Sample step of 1/2 voxel size can easily miss single bright voxels.
  # A possible fix is to compute the extrema in each voxel the slice line
  # passes through.  Use these points instead of a uniform spaced sampling.
  #
  sample_step = .5 * data_plane_spacing(v)
  steps = 1 + max(1, int(d/sample_step))

  fsteps = float(steps)
  t_list = [k/fsteps for k in range(steps)]
  xyz_list = [(1-t)*xyz_in + t*xyz_out for t in t_list]
  vertex_xform = None
  values = v.interpolated_values(xyz_list, vertex_xform, subregion = None)
  trace = zip(t_list, values)

  return trace


# -----------------------------------------------------------------------------
#
def array_slice_values(array, ijk_in, ijk_out,
                       spacing = 0.5, method = 'linear'):

  from ..geometry import vector
  d = vector.distance(ijk_in, ijk_out)
  steps = 1 + max(1, int(d/spacing))

  from numpy import empty, single as floatc, arange, outer
  trace = empty((steps,2), floatc)
  trace[:,0] = t = arange(steps, dtype = floatc) / steps
  ijk = outer(1-t, ijk_in) + outer(t, ijk_out)
  from _interpolate import interpolate_volume_data
  from ..geometry import place
  trace[:,1], outside = interpolate_volume_data(ijk, place.identity(), array, method)
  return trace

# -----------------------------------------------------------------------------
#
def data_plane_spacing(v):

  # TODO: This does not correctly compute the data plane spacing
  #       for skewed volume data.
  ijk_min, ijk_max, ijk_step = v.region
  data = v.data
  xyz_step = [a * b for a,b in zip(data.step, ijk_step)]
  spacing = min(xyz_step)
  return spacing
