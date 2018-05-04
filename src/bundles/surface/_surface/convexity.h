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
#ifndef CONVEXITY_HEADER_INCLUDED
#define CONVEXITY_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
//
// Return convexity values for each vertex defined as 2*pi - (cone angle)
// where the cone is formed by the triangles that share that vertex and
// faces toward the surface interior (counter-clockwise righthand normal
// points to exterior).  Cone angle is steradians (surface area on unit sphere).
//
// Args: vertices, triangles
// Optional args: smoothing iterations, convexity values (float64)
// Returns convexity values array (float64, same length as vertices)
//
PyObject *vertex_convexity(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
