// ----------------------------------------------------------------------------
// Find regions consisting of grid points which reach the same local maximum
// following a steepest ascent walk.
//
#include "watershed.h"

namespace Segmentation_Calculation
{

// ----------------------------------------------------------------------------
// Returns number of regions found.
// Arrays are 3-dimensional, last axis varies fastest.
//
template <class T>
Index watershed_regions(const T *data, const int *sizes,
			float threshold, Index *mask)
{
  int s0 = sizes[0], s1 = sizes[1], s2 = sizes[2];
  Index st0 = s1*s2, st1 = s2, s = s0*s1*s2;

  // Set mask to index of highest of 26 neighbors.
  for (int i0 = 0 ; i0 < s0 ; ++i0)
    {
      int j0min = (i0 > 0 ? -1 : 0), j0max = (i0+1 < s0 ? 1 : 0);
      for (int i1 = 0 ; i1 < s1 ; ++i1)
	{
	  int j1min = (i1 > 0 ? -1 : 0), j1max = (i1+1 < s1 ? 1 : 0);
	  for (int i2 = 0 ; i2 < s2 ; ++i2)
	    {
	      int j2min = (i2 > 0 ? -1 : 0), j2max = (i2+1 < s2 ? 1 : 0);
	      Index i = i0*st0 + i1*st1 + i2;
	      T v = data[i];
	      Index ni = i;
	      if (v < threshold)
		ni = 0;
	      else
		{
		  T vmax = v;
		  for (int j0 = j0min ; j0 <= j0max ; ++j0)
		    for (int j1 = j1min ; j1 <= j1max ; ++j1)
		      for (int j2 = j2min ; j2 <= j2max ; ++j2)
			{
			  Index j = i + j0*st0 + j1*st1 + j2;
			  if (data[j] > vmax)
			    { ni = j; vmax = data[j]; }
			}
		}
	      mask[i] = ni;
	    }
	}
    }
  // TODO: Handle plateaus by reconnecting neighbor chains.

  // Collapse index chains.
  for (Index i = 0 ; i < s ; ++i)
    if (mask[i] > 0)
      {
	Index ni = i;
	while (mask[ni] != ni)
	  ni = mask[ni];
	for (Index ci = i, cni ; (cni = mask[ci]) != ni ; ci = cni)
	  mask[ci] = ni;
      }

  // Renumber regions starting from 1.
  // This is a somewhat tricky algorithm that avoids using a flag bit.
  Index c = 0;
  for (Index i = 0 ; i < s ; ++i)
    if (!(data[i] < threshold))
      {
	Index mi = mask[i];
	Index mmi = mask[mi];
	if (mi < i)
	  mask[i] = mmi;	// Already renumbered region maximum.
	else if (mmi == mi)
	  {
	    mask[i] = ++c;	// Renumber region.
	    if (mi > i)
	      mask[mi] = i;	// Point region max to i.
	  }
	else
	  mask[i] = mask[mmi];
      }

  return c;		  // Return number of regions found.
}

// ----------------------------------------------------------------------------
//
template <class T>
void find_local_maxima(const T *data, const int *data_size,
		       int *start_positions, int nstart)
{
  int s0 = data_size[0], s1 = data_size[1], s2 = data_size[2];
  Index st0 = s1*s2, st1 = s2;

  for (int p = 0 ; p < nstart ; ++p)
    {
      int i2 = start_positions[3*p];
      int i1 = start_positions[3*p+1];
      int i0 = start_positions[3*p+2];
      T vmax = data[i0*st0 + i1*st1 +i2];
      int mi0 = i0, mi1 = i1, mi2 = i2;
      while (true)
	{
	  for (int o0 = -1 ; o0 <= 1 ; ++o0)
	    {
	      int ni0 = i0 + o0;
	      if (ni0 >= 0 && ni0 < s0)
		for (int o1 = -1 ; o1 <= 1 ; ++o1)
		  {
		    int ni1 = i1 + o1;
		    if (ni1 >= 0 && ni1 < s1)
		      for (int o2 = -1 ; o2 <= 1 ; ++o2)
			{
			  int ni2 = i2 + o2;
			  if (ni2 >= 0 && ni2 < s2)
			    {
			      Index ni = ni0*st0 + ni1*st1 + ni2;
			      T v = data[ni];
			      if (v > vmax)
				{
				  vmax = v;
				  mi0 = ni0; mi1 = ni1; mi2 = ni2;
				}
			    }
			}
		  }
	    }
	  if (mi0 == i0 && mi1 == i1 && mi2 == i2)
	    break;
	  else
	    { i0 = mi0; i1 = mi1; i2 = mi2; }
	}
      start_positions[3*p] = mi2;
      start_positions[3*p+1] = mi1;
      start_positions[3*p+2] = mi0;
    }
}

} // end of namespace Segmentation_Calculation
