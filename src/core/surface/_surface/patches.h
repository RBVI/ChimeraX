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

//
// Map vertex indices to unique vertex indices so vertices at the same point
// are treated as one.  This is used for connected piece calculations.
//
// Args: vertex array, n by 3 float
// Returns numpy int array, length n, mapping vertex index to unique vertex index.
//
PyObject *unique_vertex_map(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
