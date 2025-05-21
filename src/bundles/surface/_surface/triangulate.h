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
