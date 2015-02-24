// vi: set expandtab shiftwidth=4 softtabstop=4:
#ifndef DISTANCESPY_HEADER_INCLUDED
#define DISTANCESPY_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *py_distances_from_origin(PyObject *, PyObject *args);
PyObject *py_distances_perpendicular_to_axis(PyObject *, PyObject *args);
PyObject *py_distances_parallel_to_axis(PyObject *, PyObject *args);
PyObject *py_maximum_norm(PyObject *, PyObject *args, PyObject *keywds);

}

#endif
