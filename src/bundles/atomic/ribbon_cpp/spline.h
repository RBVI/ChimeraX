// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
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
