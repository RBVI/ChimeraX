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
#ifndef SPLINE_HEADER_INCLUDED
#define SPLINE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
  PyObject *natural_cubic_spline(PyObject *s, PyObject *args, PyObject *keywds);
  extern const char *natural_cubic_spline_doc;
}

#endif
