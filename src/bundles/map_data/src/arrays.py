# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
# Method can be linear or nearest.
#
def interpolate_volume_data(vertices, vertex_transform, array,
                            method = 'linear', values = None):

  kw = {} if values is None else {'values': values}
  from chimerax.map._map import interpolate_volume_data
  values, outside = interpolate_volume_data(vertices, vertex_transform.matrix,
                                              array, method, **kw)
  return values, outside
    
# -----------------------------------------------------------------------------
# Method can be linear or nearest.
#
def interpolate_volume_gradient(vertices, v2m_transform, array,
                                method = 'linear', gradients = None):

  kw = {} if gradients is None else {'gradients': gradients}
  from chimerax.map._map import interpolate_volume_gradient
  gradients, outside = interpolate_volume_gradient(vertices, v2m_transform.matrix,
                                                   array, method, **kw)
  return gradients, outside

# -----------------------------------------------------------------------------
# Minimum, maximum, and sampled matrix values.
#
class MatrixValueStatistics:

  def __init__(self, matrix, bins = 10000, ignore_pad_value = None):

    matrices = matrix if isinstance(matrix, (list, tuple)) else [matrix]

    # Determine minimum and maximum data values.
    from chimerax.map import _map
    if ignore_pad_value is None:
      mm = [_map.minimum_and_maximum(m) for m in matrices]
    else:
      mm = [_map.minimum_and_maximum(m, ignore_pad_value=ignore_pad_value)
            for m in matrices]
    if ignore_pad_value is not None:
      for (mn,mx),m in zip(mm, matrices):
        if mn > mx:
          raise ValueError(f'Computing histogram of matrix {m.shape} '
                           f'that contains only the pad value {ignore_pad_value}')
        
    self.minimum = min(mn for mn,mx in mm)
    self.maximum = max(mx for mn,mx in mm)

    # Determine center values for first and last bins, and bin size.
    fbc, lbc, bsize = self.bin_range(bins)

    # Determine count of matrix values in each bin.
    bins_start = fbc - .5*bsize
    bins_end = lbc + .5*bsize
    # TODO: Should use 64 bit integer to avoid wrapping when counting
    #  values in density maps larger than 2 Gvoxels.  Currently C++ array
    #  parsing does not support 64-bit int.
    from numpy import zeros, float64
    counts = zeros((bins,), float64)
    if ignore_pad_value is None:
      for m in matrices:
        _map.bin_counts_float64(m, bins_start, bins_end, counts)
    else:
      for m in matrices:
        _map.bin_counts_float64(m, bins_start, bins_end, counts,
                                ignore_pad_value=ignore_pad_value)
    self.counts = counts
    self.bins = bins
    self.ccounts = None         # Cumulative counts
    self.cmass = None           # Cumulative mass

  # ---------------------------------------------------------------------------
  # Return center positions of first and last bin, and bin size.
  #
  def bin_range(self, bins):

    min, max = self.minimum, self.maximum
    if bins > 1:
      if max > min:
        fbc = min
        lbc = max
      else:
        fbc = min - 1
        lbc = max + 1
      bsize = (lbc - fbc) / (bins - 1)
    else:
      fbc = lbc = .5 * (max + min)
      bsize = 2 * (max - min)
      if bsize <= 0:
        bsize = 2

    return fbc, lbc, bsize

  # ---------------------------------------------------------------------------
  # Compute histogram from binned data using a different number of bins.
  #
  def bin_counts(self, bins):

    fbins = self.bins
    fcounts = self.counts

    fbc, lbc, bsize = self.bin_range(bins)
    ffbc, flbc, fbsize = self.bin_range(fbins)
    r = bsize/fbsize
    s = 0.5 + ((fbc - ffbc) / fbsize)

    from numpy import zeros, float64, sum
    bcounts = zeros((bins,), float64)

    from math import ceil, floor
    for b in range(bins):
      fb0 = s + (b - 0.5)*r
      b0 = int(ceil(fb0))
      f0 = b0 - fb0
      if b0 < 0:
        b0 = 0
        f0 = 0
      fb1 = s + (b + 0.5)*r
      b1 = int(floor(fb1))
      f1 = fb1 - b1
      if b1 >= fbins:
        b1 = fbins
        f1 = 0
      c = 0
      if b0-1 == b1:
        c += r*fcounts[b0-1]
      else:
        if b0 > 0 and b0 <= fbins:
          c += fcounts[b0-1]*f0
        if b1 > b0:
          c += sum(fcounts[b0:b1])
        if b1 >= 0 and b1 < fbins:
          c += fcounts[b1]*f1
      bcounts[b] = c

    return bcounts

  # ---------------------------------------------------------------------------
  # Find the data value where a specified fraction of voxels have lower value.
  # Result is an approximation using binned data.
  #
  def rank_data_value(self, fraction, estimate = 'center'):
    
    ccounts = self.cumulative_counts()
    b = ccounts.searchsorted(fraction*ccounts[-1])
    fbc, lbc, bsize = self.bin_range(self.bins)
    v = fbc + b * bsize		# Center of bin where rank crossed.
    if estimate == 'high':
      v += 0.5 * bsize		# Right edge of bin where rank crossed.
    elif estimate == 'low':
      v -= 0.5 * bsize		# Left edge of bin where rank crossed.
    return v

  # ---------------------------------------------------------------------------
  # Find the fraction of voxels that hove lower data value.
  # Result is an approximation using binned data.
  #
  def data_value_rank(self, value):
    
    fbc, lbc, bsize = self.bin_range(self.bins)
    if bsize > 0:
      from math import floor
      b = int(floor((value - (fbc - 0.5 * bsize)) / bsize))
      cc = self.cumulative_counts()
      if b < 0:
        r = 0
      elif b < len(cc) and cc[-1] > 0:
        r = float(cc[b]) / cc[-1]
      else:
        r = 1.0
    elif value >= lbc:
      r = 1.0
    else:
      r = 0.0
    return r

  # ---------------------------------------------------------------------------
  #
  def cumulative_counts(self):

    if self.ccounts is None:
      self.ccounts = self.counts.cumsum()
    return self.ccounts

  # ---------------------------------------------------------------------------
  # Find the data value where the specified fraction of total mass of non-negative
  # values is below the level.  Result is an approximation using binned data.
  #
  def mass_rank_data_value(self, fraction, estimate = 'center'):
    
    cmass = self.cumulative_mass()
    fbc, lbc, bsize = self.bin_range(self.bins)
    b = cmass.searchsorted(fraction*cmass[-1])
    v = fbc + b * bsize		# Center of bin where rank crossed.
    if estimate == 'high':
      v += 0.5 * bsize		# Right edge of bin where rank crossed.
    elif estimate == 'low':
      v -= 0.5 * bsize		# Left edge of bin where rank crossed.
    return v

  # ---------------------------------------------------------------------------
  #
  def cumulative_mass(self):

    if self.cmass is None:
      fbc, lbc, bsize = self.bin_range(self.bins)
      from numpy import arange, maximum
      bin_size = (lbc-fbc)/self.bins
      vbc = arange(len(self.counts)) * bin_size + fbc
      bin_mass = maximum(self.counts * vbc, 0)
      self.cmass = bin_mass.cumsum()
    return self.cmass

# -----------------------------------------------------------------------------
#
def grid_indices(size, data_type):

  #
  # Could use numpy.indices() and swapaxes() but the following code makes
  # the array contiguous.
  #
  shape = (size[2], size[1], size[0], 3)
  from numpy import zeros, product, reshape
  indices = zeros(shape, data_type)
  for i in range(size[0]):
    indices[:,:,i,0] = i
  for j in range(size[1]):
    indices[:,j,:,1] = j
  for k in range(size[2]):
    indices[k,:,:,2] = k
  volume = product(size)
  indices = reshape(indices, (volume, 3))
  return indices

# -----------------------------------------------------------------------------
#
def zone_masked_grid_data(grid_data, zone_points, zone_radius,
                          invert_mask = False, minimal_bounds = False,
                          zone_point_mask_values = None):

  if minimal_bounds and not invert_mask:
    from .regions import points_ijk_bounds, clamp_region, integer_region
    r = points_ijk_bounds(zone_points, zone_radius, grid_data)
    r = clamp_region(integer_region(r), grid_data.size)
    from .griddata import GridSubregion
    grid_data = GridSubregion(grid_data, r[0], r[1])

  mask = zone_mask(grid_data, zone_points, zone_radius,
                   invert_mask, zone_point_mask_values)
  masked_data = masked_grid_data(grid_data, mask)

  return masked_data

# -----------------------------------------------------------------------------
#
def zone_mask(grid_data, zone_points, zone_radius,
              invert_mask = False, zone_point_mask_values = None):

  from numpy import single as floatc, array, ndarray, zeros, int8, intc

  if not isinstance(zone_points, ndarray):
    zone_points = array(zone_points, floatc)

  if (not zone_point_mask_values is None and
      not isinstance(zone_point_mask_values, ndarray)):
    zone_point_mask_values = array(zone_point_mask_values, int8)

  shape = tuple(reversed(grid_data.size))
  mask_3d = zeros(shape, int8)
  mask_1d = mask_3d.ravel()

  if zone_point_mask_values is None:
    if invert_mask:
      mask_value = 0
      mask_1d[:] = 1
    else:
      mask_value = 1

  from . import grid_indices
  from chimerax.geometry import find_closest_points

  size_limit = 2 ** 22          # 4 Mvoxels
  if mask_3d.size > size_limit:
    # Calculate plane by plane to save memory with grid point array
    xsize, ysize, zsize = grid_data.size
    grid_points = grid_indices((xsize,ysize,1), floatc)
    grid_data.ijk_to_xyz_transform.transform_points(grid_points, in_place = True)
    zstep = [grid_data.ijk_to_xyz_transform.matrix[a][2] for a in range(3)]
    for z in range(zsize):
      i1, i2, n1 = find_closest_points(grid_points, zone_points, zone_radius)
      offset = xsize*ysize*z
      if zone_point_mask_values is None:
        mask_1d[i1 + offset] = mask_value
      else:
        mask_1d[i1 + offset] = zone_point_mask_values[n1]
      grid_points[:,:] += zstep
  else:
    grid_points = grid_indices(grid_data.size, floatc)
    grid_data.ijk_to_xyz_transform.transform_points(grid_points, in_place = True)
    i1, i2, n1 = find_closest_points(grid_points, zone_points, zone_radius)
    if zone_point_mask_values is None:
      mask_1d[i1] = mask_value
    else:
      mask_1d[i1] = zone_point_mask_values[n1]

  return mask_3d

# -----------------------------------------------------------------------------
# Indices are into the flattened full data matrix.
#
def masked_grid_data(grid_data, mask, mask_value = None):

  d = grid_data
  matrix = d.full_matrix()
  from numpy import zeros, putmask
  masked = zeros(matrix.shape, matrix.dtype)
  if mask_value is None:
    putmask(masked, mask, matrix)
  else:
    putmask(masked, mask == mask_value, matrix)

  from . import ArrayGridData
  masked_grid_data = ArrayGridData(masked, d.origin, d.step,
                                   d.cell_angles, d.rotation)
  return masked_grid_data

# -----------------------------------------------------------------------------
# For signed matrix values change the sign.  For UInt8 subtract values from
# 255.  For other unsigned types subtract values from highest occuring value.
#
def invert_matrix(m):

  from numpy import array, uint8, ravel, argmax, subtract
  t = m.dtype
  signed_type = (array(-1,t) < 0)
  if signed_type:
    max = 0
  elif t.type == uint8:
    max = 255
  else:
    m1d = ravel(m)
    max = m1d[argmax(m1d)]
  subtract(array(max,t), m, m)

# -----------------------------------------------------------------------------
# Find the contour level enclosing the specified volume.
#
# The enclosed volume is a discontinuous function of the level because the
# marching cubes triangulation makes discrete changes (visually it pops) when
# the level passes through a grid point data value.
#
# This could be made much faster by starting at the level that encloses the
# correct fraction of grid points (use MatrixValueStatistics binning),
# then increase or decrease 1% to try to find a small bounding interval.
# This would require fewer subdivisions and avoid making very large surfaces
# at too low a contour level.
#
def surface_level_enclosing_volume(matrix, volume, tolerance = 1e-3,
                                   max_bisections = 30, session = None):
  from chimerax.map import _map
  l0, l1 = _map.minimum_and_maximum(matrix)
  for s in range(max_bisections):
    level = 0.5*(l0 + l1)
    try:
      varray, tarray = _map.contour_surface(matrix, level, cap_faces = True,
                                            calculate_normals = False)
    except MemoryError:
      raise MemoryError('Ran out of memory contouring at level %.3g.\n' % level)

    from chimerax.surface import enclosed_volume
    evol, holes = enclosed_volume(varray, tarray)
    if abs(evol-volume) < tolerance*volume:
      break
    if evol > volume:
      l0 = level
    else:
      l1 = level
  return level
