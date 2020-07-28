// vi: set expandtab shiftwidth=4 softtabstop=4:

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
// Compute an constant intensity surface from volume data.
//
#ifndef CONTOUR_HEADER_INCLUDED
#define CONTOUR_HEADER_INCLUDED

//
// The data values can be any of the standard C numeric types.
//
// The grid size array is in x, y, z order.
//
// The grid value for index (i0,i1,i2) where 0 <= ik < size[k] is
//
//	grid[i0*stride[0] + i1*stride[1] + i2*stride[2]]
//
// Returned vertex and triangle arrays should be freed with free_surface().
//

#include "index_types.h"	// Use GIndex, AIndex, VIndex, TIndex

namespace Contour_Calculation
{

class Contour_Surface
{
 public:
  virtual ~Contour_Surface() {};
  virtual VIndex vertex_count() = 0;
  virtual TIndex triangle_count() = 0;
  virtual void geometry(float *vertex_xyz, VIndex *triangle_vertex_indices) = 0;
  virtual void normals(float *normals) = 0;
};

template <class Data_Type>
Contour_Surface *surface(const Data_Type *grid,
			 const AIndex size[3], const GIndex stride[3],
			 float threshold, bool cap_faces);
}

#include "contour.cpp"	// template implementation


#endif
