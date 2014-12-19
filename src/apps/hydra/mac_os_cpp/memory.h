// ----------------------------------------------------------------------------
//
#ifndef MEMORY_HEADER_INCLUDED
#define MEMORY_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
PyObject *memory_size(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
