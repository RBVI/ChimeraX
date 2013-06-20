#ifndef SQUAREMESH_HEADER_INCLUDED
#define SQUAREMESH_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *principle_plane_edges(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
