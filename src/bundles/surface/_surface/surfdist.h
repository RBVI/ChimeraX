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

#ifndef SURFDIST_HEADER_INCLUDED
#define SURFDIST_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
//
// Compute the closest distance from a point to a surface.  Do this for
// each point in a list.  The distance, closest point, and side of the closest
// triangle that the given point lies on is returned in an N by 5 float32 array.
// Side +1 is the right-handed normal clockwise vertex traversal, while -1
// indicates the opposite side.  This is for determining if the given point
// is inside or outside the surface.  If a distance array (N by 5) is passed
// as an argument, it will only be modified by distances less those.  If no
// distance array is provided, a newly allocated one will be returned.
//
// Limitation: The side -1 or +1 value can be wrong if the closest point
// is at a surface cusp.  This is pretty comment when the surface is a
// volume isosurface with small connected pieces.
//
// Args: points, vertex_array, triangle_array, distances
// Return: distances
PyObject *surface_distance(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
