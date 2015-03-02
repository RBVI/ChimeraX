// ----------------------------------------------------------------------------
//
#ifndef PYPARSE_HEADER_INCLUDED
#define PYPARSE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

#include "rcarray.h"			// use FArray, IArray

bool convert_vertex_array(PyObject *vertex_array, FArray *va,
			    bool allow_copy = true, bool contiguous = true);
bool convert_triangle_array(PyObject *triangle_array, IArray *ta,
			    bool allow_copy = true, bool contiguous = true);

#endif
