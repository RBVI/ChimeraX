// vi: set expandtab shiftwidth=4 softtabstop=4:

// ----------------------------------------------------------------------------
// Compute the portion of a plane inside a given surface.
//
#ifndef TRIANGULATE_HEADER_INCLUDED
#define TRIANGULATE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

#include "border.h"			// use Vertices, Triangles, Loops

namespace Cap_Calculation
{
// Triangulates set of planar polygonal loops.
void triangulate_polygon(Loops &loops, float plane_normal[3],
			 Vertices &vertex_positions,
			 Triangles &triangle_vertex_indices);
}	// end of namespace Cap_Calculation

extern "C"
{
// triangulate_polygon(loops, normal, vertices) -> tarray
PyObject *triangulate_polygon(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
