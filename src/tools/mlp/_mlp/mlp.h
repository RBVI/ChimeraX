// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
//
#ifndef MLP_HEADER_INCLUDED
#define MLP_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *mlp_sum(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
