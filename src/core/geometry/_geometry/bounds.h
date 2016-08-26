// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
// Routines to compute bounds.
//
#ifndef BOUNDS_HEADER_INCLUDED
#define BOUNDS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// sphere_xyz_bounds(centers, radii) -> bounds
PyObject *sphere_bounds(PyObject *, PyObject *args, PyObject *keywds);
// axes_sphere_bounds(centers, radii, axes) -> axes_bounds
PyObject *sphere_axes_bounds(PyObject *, PyObject *args, PyObject *keywds);
// spheres_in_bounds(centers, radii, axes, axes_bounds, padding) -> indices
PyObject *spheres_in_bounds(PyObject *, PyObject *args, PyObject *keywds);
// bounds_overlap(bounds1, bounds2, padding) -> true/false
PyObject *bounds_overlap(PyObject *, PyObject *args, PyObject *keywds);
// points_within_planes(points, planes) -> point mask
PyObject *points_within_planes(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
