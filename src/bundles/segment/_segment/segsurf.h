// ----------------------------------------------------------------------------
// Triangulate plane intersection with a box.
//
#ifndef SEGSURF_HEADER_INCLUDED
#define SEGSURF_HEADER_INCLUDED

#include <Python.h>			// use PyObject

// ----------------------------------------------------------------------------
// Calculate surface vertices and triangles surrounding 3d region map voxels
// having a specified region index.
//
// surf_id, vertices, triangles = segment_surface(region_map, region_index)
//
extern "C" PyObject *segment_surface(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Calculate surfaces (vertices and triangles) surrounding several sets of
// 3d region map voxels.  The region map must have integer values.  The surface_ids array
// maps region index values to surface id value.
//
// segment_group_surfaces(region_map[, surface_ids]) -> list of (surface id, vertices, triangles)
//
extern "C" PyObject *segment_surfaces(PyObject *, PyObject *args, PyObject *keywds);

#endif
