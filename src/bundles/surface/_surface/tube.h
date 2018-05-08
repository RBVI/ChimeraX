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
//
#ifndef TUBE_HEADER_INCLUDED
#define TUBE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
 PyObject *tube_geometry(PyObject *s, PyObject *args, PyObject *keywds);
 PyObject *tube_geometry_colors(PyObject *s, PyObject *args, PyObject *keywds);
 PyObject *tube_triangle_mask(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
