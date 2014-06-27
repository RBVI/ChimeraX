#ifndef PARSECIF_HEADER_INCLUDED
#define PARSECIF_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *parse_mmcif_file(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
