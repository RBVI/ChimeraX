// vi: set expandtab shiftwidth=4 softtabstop=4:

// ----------------------------------------------------------------------------
// Compute the portion of a plane inside a given surface.
//
#ifndef CAPPER_HEADER_INCLUDED
#define CAPPER_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// void compute_cap(plane_normal, plane_offset, varray, tarray) -> cap_varray, cap_tarray
PyObject *compute_cap(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
