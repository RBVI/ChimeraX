// ----------------------------------------------------------------------------
//
#ifndef REPAINT_HEADER_INCLUDED
#define REPAINT_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
PyObject *repaint_window(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
