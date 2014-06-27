#ifndef PARSE_STL_HEADER_INCLUDED
#define PARSE_STL_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *parse_stl(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
