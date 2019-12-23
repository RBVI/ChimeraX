// ----------------------------------------------------------------------------
// Routines for calculationg segmentations of volume data.
//
#ifndef FLOODFILL_HEADER_INCLUDED
#define FLOODFILL_HEADER_INCLUDED

#include "mask.h"		// use Index

namespace Segmentation_Calculation
{

//
// Fill mask in connected volume region above threshold containing a
// specified point.  Only consider grid points where mask is non-zero.
// Returns number of grid points filled.
//
template <class T>
Index flood_fill(const T *data, const int *data_size,
		 int start_position[3], float threshold,
		 Index i, Index *mask);

} // end of namespace Segmentation_Calculation

#include "floodfill.cpp"	// template implementation

#endif
