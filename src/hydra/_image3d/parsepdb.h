#ifndef PARSEPDB_HEADER_INCLUDED
#define PARSEPDB_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *parse_pdb_file(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *element_radii(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
