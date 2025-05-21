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
#ifndef MEASURE_HEADER_INCLUDED
#define MEASURE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// If surface has hole then returned volume is computed by capping
// boundary loops with fans centered at geometric center of loops.
// Returns volume and hole count.
// std::pair<float,int> enclosed_volume(PyObject *vertex_array, PyObject *triangle_array)
PyObject *enclosed_volume(PyObject *s, PyObject *args, PyObject *keywds);

// Sum of triangle areas.
// float surface_area(PyObject *vertex_array, PyObject *triangle_array);
PyObject *surface_area(PyObject *s, PyObject *args, PyObject *keywds);

// Accumulate 1/3 triangle area to each vertex.
// PyObject *vertex_areas(PyObject *vertex_array, PyObject *triangle_array, PyObject *areas = NULL);
PyObject *vertex_areas(PyObject *s, PyObject *args, PyObject *keywds);

// Returns N by 2 array of vertex indices for directed edges.
// PyObject *boundary_edges(PyObject *triangle_array);
PyObject *boundary_edges(PyObject *s, PyObject *args, PyObject *keywds);

// Returns tuple of arrays of vertex indices, one array for each loop.
// PyObject *boundary_loops(PyObject *triangle_array);
PyObject *boundary_loops(PyObject *s, PyObject *args, PyObject *keywds);

// Returns unsigned char array of edge mask values with only boundary shown.
// PyObject *boundary_edge_mask(PyObject *triangle_array);
PyObject *boundary_edge_mask(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
