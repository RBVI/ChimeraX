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

#ifndef FITTINGPY_HEADER_INCLUDED
#define FITTINGPY_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *py_correlation_gradient(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_torque(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_torques(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_correlation_torque(PyObject *, PyObject *args, PyObject *keywds);
PyObject *py_correlation_torque2(PyObject *, PyObject *args, PyObject *keywds);

}

#endif
