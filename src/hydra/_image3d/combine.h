// ----------------------------------------------------------------------------
// Compute linear combination of matrices.  5x faster than numpy.
//
#ifndef COMBINE_HEADER_INCLUDED
#define COMBINE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// m = f1*m1 + f2*m2.  Requires 3-d contiguous arrays of same type.
// void linear_combination(float f1, PyObject *m1, float f2, PyObject *m2, PyObject *m);
PyObject *linear_combination(PyObject *s, PyObject *args, PyObject *keywds);

// Computes sum in 64-bit, 1-d contiguous arrays only.
// Unfortunately not available in numpy 1.4.1.
// double inner_product_64(PyObject *m1, PyObject *m2);
PyObject *inner_product_64(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
