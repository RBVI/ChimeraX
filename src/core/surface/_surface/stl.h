// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
//
#ifndef STL_HEADER_INCLUDED
#define STL_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *stl_pack(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *stl_unpack(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
