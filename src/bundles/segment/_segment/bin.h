// ----------------------------------------------------------------------------
// Routines for calculationg segmentations of volume data.
//
#ifndef BIN_HEADER_INCLUDED
#define BIN_HEADER_INCLUDED

namespace Segment_Map
{

void bin_sums(float *xyz, int n, float *v, float b0, float bsize, int bcount,
	      float *bsums, int *bcounts);

} // end of namespace Segment_Map

#endif
