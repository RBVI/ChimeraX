// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
//
#ifndef SUBDIVIDE_HEADER_INCLUDED
#define SUBDIVIDE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *subdivide_triangles(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *subdivide_mesh(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
