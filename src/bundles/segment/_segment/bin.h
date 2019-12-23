// ----------------------------------------------------------------------------
// Routines for calculationg segmentations of volume data.
//
#ifndef BIN_HEADER_INCLUDED
#define BIN_HEADER_INCLUDED

namespace Segmentation_Calculation
{

void bin_sums(float *xyz, int n, float *v, float b0, float bsize, int bcount,
	      float *bsums, int *bcounts);

} // end of namespace Segmentation_Calculation

#endif
