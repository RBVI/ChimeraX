// ----------------------------------------------------------------------------
// Triangulate plane intersection with a box.
//
#ifndef SEGSURF_HEADER_INCLUDED
#define SEGSURF_HEADER_INCLUDED

#include <Python.h>			// use PyObject

// ----------------------------------------------------------------------------
// Calculate surface vertices and triangles surrounding 3d region map voxels
// having a specified region index.  If the groups argument is given it maps
// region index to group index and the second argument refers to the group index.
//
// segmentation_surface(region_map, index [, groups]) -> (vertices, triangles)
//
extern "C" PyObject *segmentation_surface(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Calculate surface vertices and triangles for several regions of region_map.
// The region map must have integer values.  A surfce is made for each region
// integer value.  If the groups array is given it maps region index values to
// group index values and a surface is made for each group index.
//
// segmentation_surfaces(region_map[, groups]) -> list of (region_or_group_id, vertices, triangles)
//
extern "C" PyObject *segmentation_surfaces(PyObject *, PyObject *args, PyObject *keywds);

#endif
