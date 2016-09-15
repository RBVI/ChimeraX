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

#ifndef DISTANCESPY_HEADER_INCLUDED
#define DISTANCESPY_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *py_distances_from_origin(PyObject *, PyObject *args);
PyObject *py_distances_perpendicular_to_axis(PyObject *, PyObject *args);
PyObject *py_distances_parallel_to_axis(PyObject *, PyObject *args);
PyObject *py_maximum_norm(PyObject *, PyObject *args, PyObject *keywds);

}

#endif
