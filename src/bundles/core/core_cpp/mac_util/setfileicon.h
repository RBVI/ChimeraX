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
#ifndef SETFILEICON_HEADER_INCLUDED
#define SETFILEICON_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
PyObject *can_set_file_icon(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *set_file_icon(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
