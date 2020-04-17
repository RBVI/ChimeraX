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
#ifndef RSPLINE_HEADER_INCLUDED
#define RSPLINE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
  PyObject *cubic_path(PyObject *s, PyObject *args, PyObject *keywds);
  extern const char *cubic_path_doc;

  PyObject *spline_path(PyObject *s, PyObject *args, PyObject *keywds);
  extern const char *spline_path_doc;

  PyObject *cubic_spline(PyObject *s, PyObject *args, PyObject *keywds);

  PyObject *atom_spline_positions(PyObject *s, PyObject *args, PyObject *keywds);

  PyObject *set_atom_tether_positions(PyObject *s, PyObject *args, PyObject *keywds);

  PyObject *get_polymer_spline(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
