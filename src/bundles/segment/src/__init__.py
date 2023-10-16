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


# ----------------------------------------------------------------------------
# Compute watershed regions putting numeric indices for each region in region map
# array and returning the number of regions.
#
#   uint32 watershed_regions(T *data, float threshold, uint32 *region_map)
#
# ----------------------------------------------------------------------------
# Compute the integer grid indices for each region.
#
#   region_index_lists(uint32 *region_map) -> tuple of n x 3 int numpy arrays
#
# ----------------------------------------------------------------------------
# Compute region contacts returning an Nx3 array where N is the total number
# of contacting region pairs and the 3 values are (r1, r2, ncontact) with
# r1 = region1 index, r2 = region2 index, ncontact = number of contact
# between region1 and region2.
#
#   region_contacts(uint32 *region_map) -> n x 3 int numpy array
#
# ----------------------------------------------------------------------------
# Report minimum and maximum data values at boundaries between regions.
# Returns a pair of arrays, an N x 3 array giving (r1, r2, ncontact) and
# a parallel N x 2 float array giving (data_min, data_max).  Here r1 and
# r2 are the two contacting region ids, ncontact is the number of contact
# points and data_min, data_max are the minimum and maximum data values
# on the contact interface.
#
#   interface_values(uint32 *region_map, T *data) -> (nc x 3 numpy int array, nc x 2 numpy float array)
#
# ----------------------------------------------------------------------------
# Compute grid index bounds for each region.  Returns an Nx7 array where
# N = max region index + 1.  The first array index is the region number.
# The 7 values for a region are (imin, jmin, kmin, imax, jmax, kmax, npoints)
# where region points exist at the min and max values, and npoints is the
# total number of points in the region.
#
#   region_bounds(uint32 *region_map) -> n x 7 int numpy array
#
# ----------------------------------------------------------------------------
# Count the number of grid points belonging to a specified region.
#
#   uint32 region_point_count(uint32 *region_map, int region_index)
#
# ----------------------------------------------------------------------------
# Return an N x 3 array of grid indices for a specified region.
#
#   region_points(uint32 *region_map, int region_index) -> n x 3 numpy int array
#
# ----------------------------------------------------------------------------
# Report the grid index for each region where the maximum data values is attained
# and the data value at the maximum.  Two arrays are returned indexed by the
# region number (size equals maximum region index plus 1).  The first array
# is n x 3 numpy int containing (i,j,k) grid index of maximum, and the second
# array is length n 1-D numpy float array containing the maximum data value.
#
#   region_maxima(uint32 *region_map, T *data)
#     -> (n x 3 numpy int array, length n numpy float array)
#
# ----------------------------------------------------------------------------
# Find the local maxima in a 3d data array starting from specified grid points
# by travelling a steepest ascent path.  The starting points array (numpy nx3 int)
# is modified to have the grid point position of the maxima for each starting
# point.  The starting_positions array is required to be contiguous.
#
#   find_local_maxima(T *data, int *start_positions) -> None
 #
# ----------------------------------------------------------------------------
# This routine is intended to compute the midpoints along the axis of a filament
# defined as a set of region points.  The midpoints are compute over bcount intervals
# where a point p belongs to interval i = ((p,axis) - b0) / bsize (rounded down).
# Points outside the bcount intervals are not included.  Returned values are a
# bcount x 3 numpy float array giving the sum of point positions (x,y,z) in each interval,
# and a length bcount numpy int array giving the number of points in each interval.
# These can be used to compute the mean point position in each interval.
# The points array is required to be contiguous.
#  
#  crosssection_midpoints(float points[n,3], float axis[3], float b0, float bsize, int bcount)
#     -> (bcount x 3 numpy float array, length bcount int array)
#

# Make sure _segment can runtime link shared library libarrays.
import chimerax.arrays

from ._segment import watershed_regions, region_index_lists, region_contacts, region_bounds
from ._segment import region_point_count, region_points, region_maxima, interface_values
from ._segment import find_local_maxima, crosssection_midpoints

# ----------------------------------------------------------------------------
# Calculate surface vertices and triangles surrounding 3d region map voxels
# having a specified region or group index.  The optional group array maps
# region map values to group index values.
#
# segmentation_surface(region_map, index[, groups]) -> (vertices, triangles)
#
# ----------------------------------------------------------------------------
# Calculate surface vertices and triangles for several regions of region_map.
# The region map must have integer values.  A surfce is made for each region
# integer value.  If the groups array is given it maps region index values to
# group index values and a surface is made for each group index.
#
# segmentation_surfaces(region_map[, groups]) -> list of (id, vertices, triangles)
#
from ._segment import segmentation_surface, segmentation_surfaces

# -----------------------------------------------------------------------------
#
from chimerax.core.toolshed import BundleAPI

class _SegmentBundle(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        if command_name == 'segmentation':
            from . import segment
            segment.register_segmentation_command(logger)

bundle_api = _SegmentBundle()
