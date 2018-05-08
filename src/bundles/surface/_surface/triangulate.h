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
#ifndef TRIANGULATE_HEADER_INCLUDED
#define TRIANGULATE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

#include "border.h"			// use Vertices, Triangles, Loops

namespace Cap_Calculation
{
// Triangulates set of planar polygonal loops.
void triangulate_polygon(Loops &loops, float plane_normal[3],
			 Vertices &vertex_positions,
			 Triangles &triangle_vertex_indices);
}	// end of namespace Cap_Calculation

extern "C"
{
// triangulate_polygon(loops, normal, vertices) -> tarray
PyObject *triangulate_polygon(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
