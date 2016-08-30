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
#ifndef COUNT_HEADER_INCLUDED
#define COUNT_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// int count_value(uint8 1-d array, uint8 value)
PyObject *count_value(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
