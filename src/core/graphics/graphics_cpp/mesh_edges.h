#ifndef MESH_EDGES_HEADER_INCLUDED
#define MESH_EDGES_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *masked_edges(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
