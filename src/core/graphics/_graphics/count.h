// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
//
#ifndef COUNT_HEADER_INCLUDED
#define COUNT_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// int count_value(uint8 1-d array, uint8 value)
PyObject *count_value(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
