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

#ifndef GAUSSIAN_HEADER_INCLUCED
#define GAUSSIAN_HEADER_INCLUCED

extern "C"
{
PyObject *py_sum_of_gaussians(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *py_sum_of_balls(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *covariance_sum(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
