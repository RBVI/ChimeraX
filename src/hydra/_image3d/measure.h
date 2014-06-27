// ----------------------------------------------------------------------------
//
#ifndef MEASURE_HEADER_INCLUDED
#define MEASURE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// If surface has hole then returned volume is computed by capping
// boundary loops with fans centered at geometric center of loops.
// Returns volume and hole count.
// std::pair<float,int> enclosed_volume(PyObject *vertex_array, PyObject *triangle_array)
PyObject *enclosed_volume(PyObject *s, PyObject *args, PyObject *keywds);

// Sum of triangle areas.
// float surface_area(PyObject *vertex_array, PyObject *triangle_array);
PyObject *surface_area(PyObject *s, PyObject *args, PyObject *keywds);

// Accumulate 1/3 triangle area to each vertex.
// PyObject *vertex_areas(PyObject *vertex_array, PyObject *triangle_array, PyObject *areas = NULL);
PyObject *vertex_areas(PyObject *s, PyObject *args, PyObject *keywds);

// Returns N by 2 array of vertex indices for directed edges.
// PyObject *boundary_edges(PyObject *triangle_array);
PyObject *boundary_edges(PyObject *s, PyObject *args, PyObject *keywds);

// Returns tuple of arrays of vertex indices, one array for each loop.
// PyObject *boundary_loops(PyObject *triangle_array);
PyObject *boundary_loops(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
