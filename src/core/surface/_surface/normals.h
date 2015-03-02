// ----------------------------------------------------------------------------
//
#ifndef NORMALS_HEADER_INCLUDED
#define NORMALS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// Args: vertex_array, triangle_array
PyObject *calculate_vertex_normals(PyObject *s, PyObject *args, PyObject *keywds);
// Args: normals_array, triangle_array
PyObject *invert_vertex_normals(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
