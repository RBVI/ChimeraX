// ----------------------------------------------------------------------------
//
#ifndef SETFILEICON_HEADER_INCLUDED
#define SETFILEICON_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
PyObject *can_set_file_icon(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *set_file_icon(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
