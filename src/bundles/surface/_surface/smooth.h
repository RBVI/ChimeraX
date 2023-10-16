// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
//
#ifndef SMOOTH_HEADER_INCLUDED
#define SMOOTH_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
//
// Move surface vertices towards the average of neighboring vertices
// give the surface a smoother appearance.
// The vertex array is xyz points (n by 3, NumPy C float).
// The triangle array is triples of indices into the vertex array.
//
// Args: vertex_array, triangle_array, float smoothing_factor, int smoothing_iterations
//
PyObject *smooth_vertex_positions(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
