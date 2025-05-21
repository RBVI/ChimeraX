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

#include <iostream>			// use std::cerr for debugging
#include <Python.h>			// use PyObject

#include "pysegment.h"			// use watershed_regions, region_bounds, ...
#include "segsurf.h"			// use segment_surface

namespace Segment_Cpp
{

// ----------------------------------------------------------------------------
//
static struct PyMethodDef segment_methods[] =
{

  /* pysegment.h */
  {const_cast<char*>("watershed_regions"),
   (PyCFunction)watershed_regions,
   METH_VARARGS|METH_KEYWORDS,
   R"(
watershed_regions(data, threshold, region_map)

Compute watershed regions putting numeric indices for each region in region map
array and returning the number of regions.
Implemented in C++.

Parameters
----------
data : 3D array, any scalar type
threshold : float
region_map : 3d array, uint32
  This array will be filled in with region index values for each watershed region.

Returns
-------
region_count : uint32
)"
  },
  
  {const_cast<char*>("region_index_lists"),
   (PyCFunction)region_index_lists,
   METH_VARARGS|METH_KEYWORDS,
   R"(
region_index_lists(region_map)

Compute the integer grid indices for each region.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32

Returns
-------
grid_points : tuple of n x 3 array of int
)"
  },
  
  {const_cast<char*>("region_contacts"),
   (PyCFunction)region_contacts,
   METH_VARARGS|METH_KEYWORDS,
   R"(
region_contacts(region_map)
   
Compute region contacts returning an Nx3 array where N is the total number
of contacting region pairs and the 3 values are (r1, r2, ncontact) with
r1 = region1 index, r2 = region2 index, ncontact = number of contact
between region1 and region2.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32

Returns
-------
contacts : n x 3 array of int
)"
  },
  
  {const_cast<char*>("interface_values"),
   (PyCFunction)interface_values,
   METH_VARARGS|METH_KEYWORDS,
   R"(
interface_values(region_map, data)

Report minimum and maximum data values at boundaries between regions.
Returns a pair of arrays, an N x 3 array giving (r1, r2, ncontact) and
a parallel N x 2 float array giving (data_min, data_max).  Here r1 and
r2 are the two contacting region ids, ncontact is the number of contact
points and data_min, data_max are the minimum and maximum data values
on the contact interface.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32
data : 3d array, any scalar type

Returns
-------
contacts : nc x 3 array of int
data_min_max : nc x 2 array of float
)"
  },
  
  {const_cast<char*>("region_bounds"),
   (PyCFunction)region_bounds,
   METH_VARARGS|METH_KEYWORDS,
   R"(
region_bounds(region_map)

Compute grid index bounds for each region.  Returns an Nx7 array where
N = max region index + 1.  The first array index is the region number.
The 7 values for a region are (imin, jmin, kmin, imax, jmax, kmax, npoints)
where region points exist at the min and max values, and npoints is the
total number of points in the region.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32

Returns
-------
bounds : n x 7 array of int
)"
  },
  
  {const_cast<char*>("region_point_count"),
   (PyCFunction)region_point_count,
   METH_VARARGS|METH_KEYWORDS,
   R"(
region_point_count(region_map, region_index)

Count the number of grid points belonging to a specified region.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32
region_index : int

Returns
-------
count : uint32
)"
  },
  
  {const_cast<char*>("region_points"),
   (PyCFunction)region_points,
   METH_VARARGS|METH_KEYWORDS,
   R"(
region_points(region_map, region_index)

Return an N x 3 array of grid indices for a specified region.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32
region_index : int

Returns
-------
points : n x 3 array of int
)"
  },
  
  {const_cast<char*>("region_maxima"),
   (PyCFunction)region_maxima,
   METH_VARARGS|METH_KEYWORDS,
   R"(
region_maxima(region_map, data)

Report the grid index for each region where the maximum data values is attained
and the data value at the maximum.  Two arrays are returned indexed by the
region number (size equals maximum region index plus 1).  The first array
is n x 3 numpy int containing (i,j,k) grid index of maximum, and the second
array is length n 1-D numpy float array containing the maximum data value.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32
data : 3d array, any scalar type

Returns
-------
max_points : n x 3 array of int
max_values : length n array of float
)"
  },
  
  {const_cast<char*>("find_local_maxima"),
   (PyCFunction)find_local_maxima,
   METH_VARARGS|METH_KEYWORDS,
   R"(
find_local_maxima(data, start_positions)

Find the local maxima in a 3d data array starting from specified grid points
by travelling a steepest ascent path.  The starting points array (numpy nx3 int)
is modified to have the grid point position of the maxima for each starting
point.  The starting_positions array is required to be contiguous.
Implemented in C++.

Parameters
----------
data : 3d array, any scalar type
start_positions : n x 3 array of int"

Returns
-------
None
)"
  },
  
  {const_cast<char*>("crosssection_midpoints"),
   (PyCFunction)crosssection_midpoints,
   METH_VARARGS|METH_KEYWORDS,
   R"(
crosssection_midpoints(points, axis, bin_start, bin_size, bin_count)

This routine is intended to compute the midpoints along the axis of a filament
defined as a set of region points.  The midpoints are compute over bcount intervals
where a point p belongs to interval i = ((p,axis) - b0) / bsize (rounded down).
Points outside the bcount intervals are not included.  Returned values are a
bcount x 3 numpy float array giving the sum of point positions (x,y,z) in each interval,
and a length bcount numpy int array giving the number of points in each interval.
These can be used to compute the mean point position in each interval.
The points array is required to be contiguous.
Implemented in C++.

Parameters
----------
points : n x 3 array of float
axis : 3 floats
bin_start : float
bin_size : float
bin_count : int

Returns
-------
point_sums : bcount x 3 array of float
point_counts : array of int, size bcount
)"
  },

  /* segsurf.h */
  {const_cast<char*>("segmentation_surface"),
   (PyCFunction)segmentation_surface,
   METH_VARARGS|METH_KEYWORDS,
   R"(
segmentation_surface(region_map, index [, groups])

Calculate surface vertices and triangles surrounding 3d region map voxels
having a specified region or group index.  The optional group array maps
region map values to group index values.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32
index : int
  Region or group index of voxels to compute surface for.
groups : 1d array of int
  This array maps the region index to a group index allowing multiple regions to form one surface.

Returns
-------
vertices : n x 3 array of float
triangles : m x 3 array of int 
)"
  },
  
  {const_cast<char*>("segmentation_surfaces"),
   (PyCFunction)segmentation_surfaces,
   METH_VARARGS|METH_KEYWORDS,
   R"(
segmentation_surfaces(region_map [, groups])

Calculate surface vertices and triangles for several regions of region_map.
The region map must have integer values.  A surfce is made for each region
integer value.  If the groups array is given it maps region index values to
group index values and a surface is made for each group index.
Implemented in C++.

Parameters
----------
region_map : 3d array, uint32
groups : 1d array of int
  This array maps the region index to a surface id allowing multiple regions to form one surface.

Returns
-------
surfaces : list of 3-tuples (int index, vertices n x 3 array of float, triangles m x 3 array of int)
)"
  },

  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int segment_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int segment_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_segment",
        NULL,
        sizeof(struct module_state),
        segment_methods,
        NULL,
        segment_traverse,
        segment_clear,
        NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
PyMODINIT_FUNC
PyInit__segment(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    
    if (module == NULL)
      return NULL;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_segment.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}

}	// Segment_Cpp namespace
