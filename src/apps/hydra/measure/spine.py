# -----------------------------------------------------------------------------
# Calculate a path down the center of narrow tubular objects.
#
# Find the long axis (smallest momento of inertia), partition into intervals
# along that axis, find average segment grid point position in each interval,
# string those centers together.  This algorithm assumes the objects is
# reasonably straight i.e. does not backtrack (not a hairpin shape).
#
def trace_spine(region, spacing = None, tip_length = None, rgba = None):

  # Set default spacing.
  if spacing is None:
    spacing = 20.0 * min(region.segmentation.grid_step())

  # Set default tip length.
  if tip_length is None:
    tip_length = 0.2 * spacing

  # Set default color.
  if rgba is None:
    rgba = region.color

  # Find long axis.
  points = region.map_points()          # Map coordinate system.
  axis = points_long_axis(points)

  # Align bins so tip bins are of equal size.
  from numpy import dot
  pa = dot(points, axis)         # positions along axis
  b0, bsize, bcount = bin_bounds(pa.min(), pa.max(), spacing, tip_length)

  # Find average point in each bin along axis.
  bave = average_bin_point(points, axis, b0, bsize, bcount)

  # Create marker path
  mset = spine_path(region, bave, 0.1*spacing, 0.05*spacing, rgba)

  return mset

# -----------------------------------------------------------------------------
#
def points_long_axis(points):

  from numpy import ones, float32
  weights = ones((len(points),), float32)
  from Measure import inertia
  v, eval, center = inertia.moments_of_inertia([(points,weights)])
  axis = v[0]
  return axis

# -----------------------------------------------------------------------------
#
def bin_bounds(amin, amax, spacing, tip_length):

  # Align bins so tip bins are of equal size.
  alen = amax - amin
  tlen = min(tip_length, alen/4, spacing)
  bc = max(1, int((alen - 2*tlen) / spacing))
  bsize = (alen - 2*tlen) / bc
  b0 = amin + tlen - bsize
  bc += 2

  return b0, bsize, bc

# -----------------------------------------------------------------------------
#
def average_bin_point(points, v, b0, bsize, bcount):

  from _segment import bin_sums
  bsums, bcounts = bin_sums(points, v, b0, bsize, bcount)
  from numpy import nonzero
  nz = bcounts.nonzero()
  bave = bsums[nz[0],:]
  for a in (0,1,2):
    bave[:,a] /= bcounts[nz]
  return bave

# -----------------------------------------------------------------------------
#
def bin_sums(points, v, b0, bsize, bcount):

  from numpy import zeros, float32, int32, nonzero
  bsums = zeros((bcount,3), float32)
  bcounts = zeros((bcount,), int32)
  n = len(points)
  for p in range(n):
    b = int((pa[p] - b0) / bsize)
    bcounts[b] += 1
    bsums[b,:] += points[p]
  return bsums, bcounts

# -----------------------------------------------------------------------------
#
def spine_path(region, points, marker_radius, link_radius, rgba):

  from VolumePath import Marker_Set, Marker, Link
  mset = Marker_Set('Region %d spine' % region.rid)
  mset.set_transform(region.segmentation.openState.xform)
  last_marker = None
  for b, xyz in enumerate(points):
    m = Marker(mset, b+1, xyz, rgba, marker_radius)
    if last_marker:
      Link(last_marker, m, rgba, link_radius)
    last_marker = m
  return mset

# -----------------------------------------------------------------------------
#
def measure_diameter(region, spine, outline_model = None):

  # Find mid-point of spine curve and tangent.
  p, t = path_midpoint(spine)
  if p is None:
    return None, None

  # Find region points in slab centered mid-point, perpendicular to tangent,
  #  with thickness = 3 times max grid spacing.
  thickness = 3 * max(region.segmentation.grid_step())
  points = slab_points(region.map_points(), t, p, thickness)  # Map coord system
  if len(points) == 0:
    return None, None

  # TODO: Don't assume spine model aligned to segmentation
  
  # Find diameters along perpendicular axes spaced at 10 degree steps.
  # Record max/min diameters as region attribute.
  import Matrix
  x,y,z = Matrix.orthonormal_frame(t)
  x1, y1, xrange, yrange = minimum_area_box(points, x, y)
  dx1, dy1 = xrange[1] - xrange[0], yrange[1] - yrange[0]
  dmax = max(dx1, dy1)
  dmin = min(dx1, dy1)

  # Optionally Display max/min diameters using two boxes.
  if outline_model:
    from numpy import dot
    pz = dot(p,z)
    zrange = (pz - 0.5*thickness, pz + 0.5*thickness)
    bounds = (xrange, yrange, zrange)
    import PickBlobs
    s = PickBlobs.outline_box_surface((x1,y1,z), bounds, None, outline_model)

  return dmax, dmin

# -----------------------------------------------------------------------------
#
def path_midpoint(mset):

  # TODO: This code belongs with the spline code in VolumePath.
  from VolumePath import tube, spline
  chains = tube.atom_chains([m.atom for m in mset.markers()])
  if len(chains) == 0:
    return None, None
  achain = chains[0][0]
  from numpy import array
  xyz_path = array([a.coord().data() for a in achain])
  alengths = spline.arc_lengths(xyz_path)
  d = 0.5 * alengths[-1]
  for i,a in enumerate(alengths):
    if a > d:
      break
  f = (d-alengths[i-1])/(alengths[i]-alengths[i-1])
  import Matrix
  p = Matrix.linear_combination(1-f, xyz_path[i-1], f, xyz_path[i])
  t = Matrix.linear_combination(1, xyz_path[i], -1, xyz_path[i-1])
  tangent = Matrix.normalize_vector(t)
  return p, tangent

# -----------------------------------------------------------------------------
#
def slab_points(points, normal, p0, thickness):

  from numpy import dot, logical_and
  delta = dot(points,normal)
  dp = dot(p0,normal)
  dmin = dp - 0.5*thickness
  dmax = dp + 0.5*thickness
  in_slab = logical_and(delta >= dmin, delta <= dmax)
  sp = points[in_slab.nonzero()[0],:]
  return sp

# -----------------------------------------------------------------------------
#
def minimum_area_box(points, x, y, nangles = 18):

  from numpy import array, dot
  x, y = array(x), array(y)
  from math import cos, sin
  amin = None
  for i in range(nangles):
    a = i*90.0/nangles
    s, c = sin(a), cos(a)
    x1 = c*x + s*y
    y1 = -s*x + c*y
    px1, py1 = dot(points,x1), dot(points,y1)
    xrange, yrange = (px1.min(), px1.max()), (py1.min(), py1.max())
    a = (xrange[1]-xrange[0])*(yrange[1]-yrange[0])
    if amin is None or a < amin:
      amin = a
      r = x1, y1, xrange, yrange
  return r

# ------------------------------------------------------------------------------
# Compute min, max, and average curvature for selected path atoms.
#
def measure_curvature(mset):

    atoms = [m.atom for m in mset.markers()]
    a2 = [a for a in atoms if len(a.bonds) == 2]

    kmax = kmin = None
    ksum = 0
    for a in a2:
        b1, b2 = a.bonds
        s = 0.5 * (b1.length() + b2.length())
        from chimera import angle
        t = angle(a.coord() - b1.otherAtom(a).coord(),
                  b2.otherAtom(a).coord() - a.coord())
        from math import pi
        trad = t*pi/180
        k = trad/s
        if kmin is None or k < kmin:
            kmin = k
        if kmax is None or k > kmax:
            kmax = k
        ksum += k
    kave = ksum / len(a2) if len(a2) > 0 else None
    return kave, kmin, kmax

# -----------------------------------------------------------------------------
#
def test(spacing = 200.0, tip_length = 50):

  from Segger.regions import Segmentation
  from chimera import selection
  segs = [m for m in selection.currentGraphs() if isinstance(m, Segmentation)]
  for seg in segs:
    for r in seg.selected_regions():
      mset = trace_spine(r, spacing, tip_length, r.color)
      mset.region = r
