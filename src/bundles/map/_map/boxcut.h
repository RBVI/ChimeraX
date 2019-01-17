// ----------------------------------------------------------------------------
// Triangulate plane intersection with a box.
//
#ifndef BOXCUT_HEADER_INCLUDED
#define BOXCUT_HEADER_INCLUDED

#include <Python.h>			// use PyObject

namespace Map_Cpp
{

extern "C"
{

// ----------------------------------------------------------------------------
// Triangulate plane intersection with a box at a sequence of equally spaced
// positions perpendicular to an axis.  Eight corners must be in order
// {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}
// Returned vertex array must be at least size 7*3*num_cuts and
// triangle array must be at least size 6*3*num_cuts.
// Returns correctly sized vertex and triangle arrays that share memory
// with input vertex and triangle arrays if provided.
//
// void box_cuts(const float corners[8][3], const float axis[3],
//               float offset, float spacing, int num_cuts,
//               float *vertices, int *nv, int *triangles, int *nt);
//
PyObject *box_cuts(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Find min/max position of corners along an axis.
//
// void offset_range(const float corners[8][3], const float axis[3],
//		     float *offset_min, float *offset_max);
//
PyObject *offset_range(PyObject *, PyObject *args, PyObject *keywds);

}	// end extern C

}	// end of namespace Map_Cpp

#endif
