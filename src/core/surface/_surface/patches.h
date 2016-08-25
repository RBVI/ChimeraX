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
