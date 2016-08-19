// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Routines to compute rotation of standard cylinder aligned along z axis
// to have specified end points.  For drawing molecule bonds.
//
#ifndef CYLINDERROT_HEADER_INCLUDED
#define CYLINDERROT_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// cylinder_rotations(xyz0, xyz1, radii, rot44)
PyObject *cylinder_rotations(PyObject *, PyObject *args, PyObject *keywds);
// cylinder_rotations_x3d(xyz0, xyz1, radii, float [n * 9])
PyObject *cylinder_rotations_x3d(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
