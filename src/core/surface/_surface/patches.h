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
// Args: vertices, normals, triangles, vertex to atom index array, atom positions.
// Returns subdivide vertices, normals, triangles, vertex to atom
//
PyObject *sharp_edge_patches(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
