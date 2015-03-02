// ----------------------------------------------------------------------------
//
#ifndef NORMALS_HEADER_INCLUDED
#define NORMALS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

#include "rcarray.h"			// use FArray, IArray

FArray calculate_vertex_normals(const FArray &vertices, const IArray &triangles);
void invert_vertex_normals(const FArray &normals, const IArray &triangles);

extern "C"
{
// Args: vertex_array, triangle_array
PyObject *calculate_vertex_normals(PyObject *s, PyObject *args, PyObject *keywds);
// Args: normals_array, triangle_array
PyObject *invert_vertex_normals(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
