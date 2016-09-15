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
#ifndef NORMALS_HEADER_INCLUDED
#define NORMALS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// Args: vertex_array, triangle_array
PyObject *calculate_vertex_normals(PyObject *s, PyObject *args, PyObject *keywds);
// Args: normals_array, triangle_array
PyObject *invert_vertex_normals(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
