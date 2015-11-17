// ----------------------------------------------------------------------------
// Compute the intersection of a surface with a plane.
//
#ifndef BORDER_HEADER_INCLUDED
#define BORDER_HEADER_INCLUDED

#include <utility>			// use std::pair
#include <vector>			// use std::vector

#include "rcarray.h"			// use FArray, IArray

namespace Cap_Calculation
{
typedef std::vector<float> Vertices;
typedef std::vector<int> Triangles;
typedef std::pair<int,int> Loop;
typedef std::vector<Loop> Loops;

// Finds plane intersection with surface as set of polygonal loops.
void calculate_border(float plane_normal[3], float plane_offset,
		      const FArray &varray, const IArray &tarray, /* Surface */
		      Vertices &border_vertices, Loops &loops);

}	// end of namespace Cap_Calculation

#endif
