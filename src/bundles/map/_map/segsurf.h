// ----------------------------------------------------------------------------
// Triangulate plane intersection with a box.
//
#ifndef SEGSURF_HEADER_INCLUDED
#define SEGSURF_HEADER_INCLUDED

#include <Python.h>			// use PyObject

// ----------------------------------------------------------------------------
// Calculate surface vertices and triangles surrounding 3d image voxels
// having a specified value.
//
// vertices, triangles = segment_surface(image3d, value)
//
extern "C" PyObject *segment_surface(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Calculate surfaces (vertices and triangles) surrounding each set of
// 3d image voxels having one voxel value.
//
// segment_surfaces(image3d) -> list of (image value, vertices, triangles)
//
extern "C" PyObject *segment_surfaces(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Calculate surfaces (vertices and triangles) surrounding several sets of
// 3d image voxels.  The image must have integer values.  The surface_ids array
// maps integer image values to surface id value.
//
// segment_group_surfaces(image3d, surface_ids) -> list of (surface id, vertices, triangles)
//
extern "C" PyObject *segment_group_surfaces(PyObject *, PyObject *args, PyObject *keywds);

#endif
