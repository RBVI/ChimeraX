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
