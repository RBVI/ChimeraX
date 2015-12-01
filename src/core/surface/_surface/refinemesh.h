// vi: set expandtab shiftwidth=4 softtabstop=4:

// ----------------------------------------------------------------------------
// Modify a planar surface triangulation to create uniform size triangles
// suited for vertex coloring.
//
#ifndef REFINEMESH_HEADER_INCLUDED
#define REFINEMESH_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// refine_mesh(vertices, triangles, subdivision_factor) -> ref_vertices, ref_triangles
PyObject *refine_mesh(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
