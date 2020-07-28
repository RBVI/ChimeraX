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
#ifndef FILLED_RING_HEADER_INCLUDED
#define FILLED_RING_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// return small (3, 4, or 5 member) ring geometry
//   fill_small_ring(float varray[n,3], float offset)
//     -> vertices, normals, triangles
PyObject *fill_small_ring(PyObject *s, PyObject *args, PyObject *keywds);
extern const char *fill_small_ring_doc;

// return 6 member ring geometry
//   fill_6ring(float varray[n,3], float offset, int anchor_corner)
//     -> vertices, normals, triangles
PyObject *fill_6ring(PyObject *s, PyObject *args, PyObject *keywds);
extern const char *fill_6ring_doc;

}

#endif
