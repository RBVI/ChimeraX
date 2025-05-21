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
#ifndef PATCHES_HEADER_INCLUDED
#define PATCHES_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
//
// Split triangles to create sharp boundaries equidistant between atoms.
//
// Args: vertices, normals, triangles, vertex to atom index array, atom positions, atom radius.
// Returns subdivide vertices, normals, triangles, triangles without duplicate vertices, vertex to atom
//
PyObject *sharp_edge_patches(PyObject *, PyObject *args, PyObject *keywds);

//
// Map vertex indices to unique vertex indices so vertices at the same point
// are treated as one.  This is used for connected piece calculations.
// This calculation is quite slow, 2 million vertices per second.
//
// Args: vertex array, n by 3 float
// Returns numpy int array, length n, mapping vertex index to unique vertex index.
//
PyObject *unique_vertex_map(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
