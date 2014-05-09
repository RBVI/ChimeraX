// ----------------------------------------------------------------------------
//
#ifndef TUBE_HEADER_INCLUDED
#define TUBE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
 PyObject *tube_geometry(PyObject *s, PyObject *args, PyObject *keywds);
 PyObject *tube_geometry_colors(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
