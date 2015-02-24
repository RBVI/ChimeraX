// vi: set expandtab shiftwidth=4 softtabstop=4:
#ifndef HISTOGRAM_HEADER_INCLUDED
#define HISTOGRAM_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *minimum_and_maximum(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *bin_counts_py(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *high_count_py(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *high_indices_py(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
