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
// Compute the portion of a plane inside a given surface.
//
#ifndef CAPPER_HEADER_INCLUDED
#define CAPPER_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// void compute_cap(plane_normal, plane_offset, varray, tarray) -> cap_varray, cap_tarray
PyObject *compute_cap(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
