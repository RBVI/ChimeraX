// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
//
#ifndef SPLINE_HEADER_INCLUDED
#define SPLINE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
 PyObject *natural_cubic_spline(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
