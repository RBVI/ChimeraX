#ifndef DISTGRID_HEADER_INCLUDED
#define DISTGRID_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
PyObject *py_sphere_surface_distance(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
