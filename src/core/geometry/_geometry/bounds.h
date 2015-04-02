// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Routines to compute bounds.
//
#ifndef BOUNDS_HEADER_INCLUDED
#define BOUNDS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// axes_sphere_bounds(centers, radii, axes) -> axes_bounds
PyObject *sphere_bounds(PyObject *, PyObject *args, PyObject *keywds);
// spheres_in_bounds(centers, radii, axes, axes_bounds, padding) -> indices
PyObject *spheres_in_bounds(PyObject *, PyObject *args, PyObject *keywds);
// bounds_overlap(bounds1, bounds2, padding) -> true/false
PyObject *bounds_overlap(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
