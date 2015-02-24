// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Compute linear combination of vectors.  5x faster than numpy.
//
#ifndef VECTOR_OPS_HEADER_INCLUDED
#define VECTOR_OPS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// Computes sum in 64-bit, 1-d arrays only.
PyObject *inner_product_64(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
