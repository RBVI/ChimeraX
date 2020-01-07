// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
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
static struct PyMethodDef segment_cpp_methods[] =
{

  /* pysegment.h */
  {const_cast<char*>("watershed_regions"), (PyCFunction)watershed_regions,
   METH_VARARGS|METH_KEYWORDS,
   "watershed_regions(data, threshold, region_map)\n"
   "\n"
   "Compute watershed regions putting numeric indices for each region in region map\n"
   "array and returning the number of regions.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "data : 3D array, any scalar type\n"
   "threshold : float\n"
   "region_map : 3d array, uint32\n"
   "  This array will be filled in with region index values for each watershed region.\n"
   "\n"
   "Returns\n"
   "-------\n"
   "region_count : uint32\n"
  },
  {const_cast<char*>("region_index_lists"), (PyCFunction)region_index_lists,
   METH_VARARGS|METH_KEYWORDS,
   "region_index_lists(region_map)\n"
   "\n"
   "Compute the integer grid indices for each region.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "\n"
   "Returns\n"
   "-------\n"
   "grid_points : tuple of n x 3 array of int\n"
  },
  {const_cast<char*>("region_contacts"), (PyCFunction)region_contacts,
   METH_VARARGS|METH_KEYWORDS,
   "region_contacts(region_map)\n"
   "\n"   
   "Compute region contacts returning an Nx3 array where N is the total number\n"
   "of contacting region pairs and the 3 values are (r1, r2, ncontact) with\n"
   "r1 = region1 index, r2 = region2 index, ncontact = number of contact\n"
   "between region1 and region2.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "\n"
   "Returns\n"
   "-------\n"
   "contacts : n x 3 array of int\n"
  },
  {const_cast<char*>("interface_values"), (PyCFunction)interface_values,
   METH_VARARGS|METH_KEYWORDS,
   "interface_values(region_map, data)\n"
   "\n"
   "Report minimum and maximum data values at boundaries between regions.\n"
   "Returns a pair of arrays, an N x 3 array giving (r1, r2, ncontact) and\n"
   "a parallel N x 2 float array giving (data_min, data_max).  Here r1 and\n"
   "r2 are the two contacting region ids, ncontact is the number of contact\n"
   "points and data_min, data_max are the minimum and maximum data values\n"
   "on the contact interface.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "data : 3d array, any scalar type\n"
   "\n"
   "Returns\n"
   "-------\n"
   "interface_stats : (nc x 3 array of int, nc x 2 array of float)\n"
  },
  {const_cast<char*>("region_bounds"), (PyCFunction)region_bounds,
   METH_VARARGS|METH_KEYWORDS,
   "region_bounds(region_map)\n"
   "\n"
   "Compute grid index bounds for each region.  Returns an Nx7 array where\n"
   "N = max region index + 1.  The first array index is the region number.\n"
   "The 7 values for a region are (imin, jmin, kmin, imax, jmax, kmax, npoints)\n"
   "where region points exist at the min and max values, and npoints is the\n"
   "total number of points in the region.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "\n"
   "Returns\n"
   "-------\n"
   "bounds : n x 7 array of int\n"
  },
  {const_cast<char*>("region_point_count"), (PyCFunction)region_point_count,
   METH_VARARGS|METH_KEYWORDS,
   "region_point_count(region_map, region_index)\n"
   "\n"
   "Count the number of grid points belonging to a specified region.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "region_index : int\n"
   "\n"
   "Returns\n"
   "-------\n"
   "count : uint32\n"
  },
  {const_cast<char*>("region_points"), (PyCFunction)region_points,
   METH_VARARGS|METH_KEYWORDS,
   "region_points(region_map, region_index)\n"
   "\n"
   "Return an N x 3 array of grid indices for a specified region.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "region_index : int\n"
   "\n"
   "Returns\n"
   "-------\n"
   "points : n x 3 array of int\n"
  },
  {const_cast<char*>("region_maxima"), (PyCFunction)region_maxima,
   METH_VARARGS|METH_KEYWORDS,
   "region_maxima(region_map, data)\n"
   "\n"
   "Report the grid index for each region where the maximum data values is attained\n"
   "and the data value at the maximum.  Two arrays are returned indexed by the\n"
   "region number (size equals maximum region index plus 1).  The first array\n"
   "is n x 3 numpy int containing (i,j,k) grid index of maximum, and the second\n"
   "array is length n 1-D numpy float array containing the maximum data value.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "data : 3d array, any scalar type\n"
   "\n"
   "Returns\n"
   "-------\n"
   "maxima : (n x 3 array of int, length n array of float)\n"
  },
  {const_cast<char*>("find_local_maxima"), (PyCFunction)find_local_maxima,
   METH_VARARGS|METH_KEYWORDS,
   "find_local_maxima(data, start_positions)\n"
   "\n"
   "Find the local maxima in a 3d data array starting from specified grid points\n"
   "by travelling a steepest ascent path.  The starting points array (numpy nx3 int)\n"
   "is modified to have the grid point position of the maxima for each starting\n"
   "point.  The starting_positions array is required to be contiguous.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "data : 3d array, any scalar type\n"
   "start_positions : n x 3 array of int"
   "\n"
   "Returns\n"
   "-------\n"
   "None\n"
  },
  {const_cast<char*>("crosssection_midpoints"), (PyCFunction)crosssection_midpoints,
   METH_VARARGS|METH_KEYWORDS,
   "crosssection_midpoints(points, axis, bin_start, bin_size, bin_count)\n"
   "\n"
   "This routine is intended to compute the midpoints along the axis of a filament\n"
   "defined as a set of region points.  The midpoints are compute over bcount intervals\n"
   "where a point p belongs to interval i = ((p,axis) - b0) / bsize (rounded down).\n"
   "Points outside the bcount intervals are not included.  Returned values are a\n"
   "bcount x 3 numpy float array giving the sum of point positions (x,y,z) in each interval,\n"
   "and a length bcount numpy int array giving the number of points in each interval.\n"
   "These can be used to compute the mean point position in each interval.\n"
   "The points array is required to be contiguous.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "points : n x 3 array of float\n"
   "axis : 3 floats\n"
   "bin_start : float\n"
   "bin_size : float\n"
   "bin_count : int\n"
   "\n"
   "Returns\n"
   "-------\n"
   "midpoints : (bcount x 3 array of float, length bcount array of int)\n"
  },

  /* segsurf.h */
  {const_cast<char*>("segment_surface"), (PyCFunction)segment_surface,
   METH_VARARGS|METH_KEYWORDS,
   "segment_surface(region_map, region_index)\n"
   "\n"
   "Calculate surface vertices and triangles surrounding 3d region map voxels\n"
   "having a specified region index.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "region_index : int\n"
   "\n"
   "Returns\n"
   "-------\n"
   "surface : (vertices n x 3 array of float, triangles m x 3 array of int)\n" 
  },
  {const_cast<char*>("segment_surfaces"), (PyCFunction)segment_surfaces,
   METH_VARARGS|METH_KEYWORDS,
   "segment_surfaces(region_map)\n"
   "\n"
   "Calculate surfaces (vertices and triangles) surrounding each set of\n"
   "3d region map voxels having the same region index value.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "\n"
   "Returns\n"
   "-------\n"
   "surfaces : list of 3-tuples (int region_index, vertices n x 3 array of float, triangles m x 3 array of int)\n"
  },
  {const_cast<char*>("segment_group_surfaces"), (PyCFunction)segment_group_surfaces,
   METH_VARARGS|METH_KEYWORDS,
   "segment_group_surfaces(region_map, surface_ids)\n"
   "\n"
   "Calculate surfaces (vertices and triangles) surrounding several sets of\n"
   "3d region map voxels.  The region map must have integer values.  The surface_ids array\n"
   "maps region index values to surface id value.\n"
   "Implemented in C++.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "region_map : 3d array, uint32\n"
   "surface_ids : 1d array of int\n"
   "  This array maps the region index to a surface id allowing multiple regions to be grouped forming one surface.\n"
   "\n"
   "Returns\n"
   "-------\n"
   "surfaces : list of 3-tuples (int surface_id, vertices n x 3 array of float, triangles m x 3 array of int)\n"
  },

  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int segment_cpp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int segment_cpp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "segment_cpp",
        NULL,
        sizeof(struct module_state),
        segment_cpp_methods,
        NULL,
        segment_cpp_traverse,
        segment_cpp_clear,
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

    st->error = PyErr_NewException("segment_cpp.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}

}	// Segment_Cpp namespace
