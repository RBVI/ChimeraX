// ----------------------------------------------------------------------------
// Routines for calculationg segmentations of volume data.
//
#ifndef WATERSHED_HEADER_INCLUDED
#define WATERSHED_HEADER_INCLUDED

#include "mask.h"		// use Index

namespace Segmentation_Calculation
{

//
// Find regions consisting of grid points above specified threshold
// which reach the same local maximum taking a steepest ascent walk.
// Calculates index mask and returns number of regions found.
//
template <class T>
Index watershed_regions(const T *data, const int *data_size,
			float threshold, Index *mask);

//
// Do a steepest ascent walk from given starting index positions until
// a local maximum is reached.  The positions array is modified to hold
// the index positions of the maxima.
//
template <class T>
void find_local_maxima(const T *data, const int *data_size,
		       int *start_positions, int nstart);

} // end of namespace Segmentation_Calculation

#include "watershed.cpp"	// template implementation

#endif
