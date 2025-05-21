// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <Python.h>			// use PyObject

namespace Segment_Cpp
{

extern "C"
{

// ----------------------------------------------------------------------------
// Compute watershed regions putting numeric indices for each region in region map
// array and returning the number of regions.
//
//   uint32 watershed_regions(T *data, float threshold, uint32 *region_map)
//
PyObject *watershed_regions(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Compute the integer grid indices for each region.
//
//   region_index_lists(uint32 *region_map) -> tuple of n x 3 int numpy arrays
//
PyObject *region_index_lists(PyObject *, PyObject *args, PyObject *keywds);
  
// ----------------------------------------------------------------------------
// Compute region contacts returning an Nx3 array where N is the total number
// of contacting region pairs and the 3 values are (r1, r2, ncontact) with
// r1 = region1 index, r2 = region2 index, ncontact = number of contact
// between region1 and region2.
//
//   region_contacts(uint32 *region_map) -> n x 3 int numpy array
//
PyObject *region_contacts(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Report minimum and maximum data values at boundaries between regions.
// Returns a pair of arrays, an N x 3 array giving (r1, r2, ncontact) and
// a parallel N x 2 float array giving (data_min, data_max).  Here r1 and
// r2 are the two contacting region ids, ncontact is the number of contact
// points and data_min, data_max are the minimum and maximum data values
// on the contact interface.
//
//   interface_values(uint32 *region_map, T *data) -> (nc x 3 numpy int array, nc x 2 numpy float array)
//
PyObject *interface_values(PyObject *, PyObject *args, PyObject *keywds);
  
// ----------------------------------------------------------------------------
// Compute grid index bounds for each region.  Returns an Nx7 array where
// N = max region index + 1.  The first array index is the region number.
// The 7 values for a region are (imin, jmin, kmin, imax, jmax, kmax, npoints)
// where region points exist at the min and max values, and npoints is the
// total number of points in the region.
//
//   region_bounds(uint32 *region_map) -> n x 7 int numpy array
//
PyObject *region_bounds(PyObject *, PyObject *args, PyObject *keywds);
  
// ----------------------------------------------------------------------------
// Count the number of grid points belonging to a specified region.
//
//   uint32 region_point_count(uint32 *region_map, int region_index)
//
PyObject *region_point_count(PyObject *, PyObject *args, PyObject *keywds);
  
// ----------------------------------------------------------------------------
// Return an N x 3 array of grid indices for a specified region.
//
//   region_points(uint32 *region_map, int region_index) -> n x 3 numpy int array
//
PyObject *region_points(PyObject *, PyObject *args, PyObject *keywds);
  
// ----------------------------------------------------------------------------
// Report the grid index for each region where the maximum data values is attained
// and the data value at the maximum.  Two arrays are returned indexed by the
// region number (size equals maximum region index plus 1).  The first array
// is n x 3 numpy int containing (i,j,k) grid index of maximum, and the second
// array is length n 1-D numpy float array containing the maximum data value.
//
//   region_maxima(uint32 *region_map, T *data)
//     -> (n x 3 numpy int array, length n numpy float array)
//
PyObject *region_maxima(PyObject *, PyObject *args, PyObject *keywds);
  
// ----------------------------------------------------------------------------
// Find the local maxima in a 3d data array starting from specified grid points
// by travelling a steepest ascent path.  The starting points array (numpy nx3 int)
// is modified to have the grid point position of the maxima for each starting
// point.  The starting_positions array is required to be contiguous.
//
//   find_local_maxima(T *data, int *start_positions) -> None
//
PyObject *find_local_maxima(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// This routine is intended to compute the midpoints along the axis of a filament
// defined as a set of region points.  The midpoints are compute over bcount intervals
// where a point p belongs to interval i = ((p,axis) - b0) / bsize (rounded down).
// Points outside the bcount intervals are not included.  Returned values are a
// bcount x 3 numpy float array giving the sum of point positions (x,y,z) in each interval,
// and a length bcount numpy int array giving the number of points in each interval.
// These can be used to compute the mean point position in each interval.
// The points array is required to be contiguous.
//  
//  crosssection_midpoints(float points[n,3], float axis[3], float b0, float bsize, int bcount)
//     -> (bcount x 3 numpy float array, length bcount int array)
//
PyObject *crosssection_midpoints(PyObject *, PyObject *args, PyObject *keywds);

}	// end extern C

}	// end of namespace Segment_Cpp
