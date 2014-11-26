#ifndef ARRAYOPS_HEADER_INCLUDED
#define ARRAYOPS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *value_ranges(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *contiguous_intervals(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *mask_intervals(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *duplicate_midpoints(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
