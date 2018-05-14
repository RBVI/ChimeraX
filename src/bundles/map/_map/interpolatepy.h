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

#ifndef PYINTERPOLATE_HEADER_INCLUDED
#define PYINTERPOLATE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *interpolate_volume_data(PyObject *, PyObject *args);
PyObject *interpolate_volume_gradient(PyObject *, PyObject *args);
PyObject *interpolate_colormap(PyObject *, PyObject *args);
PyObject *set_outside_volume_colors(PyObject *, PyObject *args);

}

#endif
