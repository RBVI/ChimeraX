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
#ifndef CONNECTED_HEADER_INCLUDED
#define CONNECTED_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
//
// Return sorted array of triangle indices of triangles connected to the
// specified triangle.  Two triangles are connected if they share a vertex.
// The surface must be oriented and at most two triangles can share an edge.
// The triangle array is triples of indices of vertices (m by 3, Numpy int32).
//
// Args: triangle_array, int tindex
PyObject *connected_triangles(PyObject *s, PyObject *args, PyObject *keywds);
// Args: triangle_array, triangle_list (int array)
PyObject *triangle_vertices(PyObject *s, PyObject *args, PyObject *keywds);

//
// Return each connected piece of a surface as a separate triangle array
// and vertex array.  The return value is a tuple of pairs of vertex and
// triangle index arrays.  Vertices connected by any sequence of triangle
// edges are considered connected.
//
// Args: triangle_array (N by 3 int)
PyObject *connected_pieces(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
