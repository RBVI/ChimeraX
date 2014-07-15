// ----------------------------------------------------------------------------
// Compute bounds of a set of points.
//
#ifndef BOUNDS_HEADER_INCLUDED
#define BOUNDS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

PyObject *point_bounds(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
