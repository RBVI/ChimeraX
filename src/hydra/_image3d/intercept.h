// ----------------------------------------------------------------------------
//
#ifndef INTERCEPT_HEADER_INCLUDED
#define INTERCEPT_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
 PyObject *closest_geometry_intercept(PyObject *s, PyObject *args, PyObject *keywds);
 PyObject *closest_sphere_intercept(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
