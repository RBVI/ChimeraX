// ----------------------------------------------------------------------------
// Routines for calculationg segmentations of volume data.
//
#ifndef FLOODFILL_HEADER_INCLUDED
#define FLOODFILL_HEADER_INCLUDED

#include "region_map.h"		// use Index

namespace Segment_Map
{

//
// Fill region map in connected volume region above threshold containing a
// specified point.  Only consider grid points where region map is non-zero.
// Returns number of grid points filled.
//
template <class T>
Index flood_fill(const T *data, const int *data_size,
		 int start_position[3], float threshold,
		 Index i, Index *region_map);

} // end of namespace Segment_Map

#include "floodfill.cpp"	// template implementation

#endif
