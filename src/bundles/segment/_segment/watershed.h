// ----------------------------------------------------------------------------
// Routines for calculationg segmentations of volume data.
//
#ifndef WATERSHED_HEADER_INCLUDED
#define WATERSHED_HEADER_INCLUDED

#include "region_map.h"		// use Index

namespace Segment_Map
{

//
// Find regions consisting of grid points above specified threshold
// which reach the same local maximum taking a steepest ascent walk.
// Calculates region map and returns number of regions found.
//
template <class T>
Index watershed_regions(const T *data, const int64_t *data_size,
			float threshold, Index *region_map);

//
// Do a steepest ascent walk from given starting index positions until
// a local maximum is reached.  The positions array is modified to hold
// the index positions of the maxima.
//
template <class T>
void find_local_maxima(const T *data, const int64_t *data_size,
		       int *start_positions, int nstart);

} // end of namespace Segment_Map

#include "watershed.cpp"	// template implementation

#endif
