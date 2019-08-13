// ----------------------------------------------------------------------------
//
#include <vector>		// use std::vector

#include "floodfill.h"

namespace Segmentation_Calculation
{

class Grid_Index
{
public:
  Grid_Index(int i0, int i1, int i2) : i0(i0), i1(i1), i2(i2) {}
  int i0, i1, i2;
};

// ----------------------------------------------------------------------------
//
template <class T>
Index flood_fill(const T *data, const int *data_size,
		 int start_position[3], float threshold,
		 Index mv, Index *mask)
{
  int s0 = data_size[0], s1 = data_size[1], s2 = data_size[2];
  Index st0 = s1*s2, st1 = s2;

  std::vector<Grid_Index> bndry;

  int i0 = start_position[2], i1 = start_position[1], i2 = start_position[0];
  Index i = i0*st0 + i1*st1 + i2;
  if (mask[i] == 0 && data[i] >= threshold)
    bndry.push_back(Grid_Index(i0,i1,i2));

  Index c = 0;
  while (bndry.size() > 0)
    {
      Grid_Index p = bndry.back();
      bndry.pop_back();
      i0 = p.i0; i1 = p.i1; i2 = p.i2;
      Index i = i0*st0 + i1*st1 + i2;
      if (mask[i] == 0)
	{
	  mask[i] = mv;
	  c += 1;
	  if (i0 > 0 && mask[i-st0] == 0 && data[i-st0] >= threshold)
	    bndry.push_back(Grid_Index(i0-1,i1,i2));
	  if (i1 > 0 && mask[i-st1] == 0 && data[i-st1] >= threshold)
	    bndry.push_back(Grid_Index(i0,i1-1,i2));
	  if (i2 > 0 && mask[i-1] == 0 && data[i-1] >= threshold)
	    bndry.push_back(Grid_Index(i0,i1,i2-1));
	  if (i0+1 < s0 && mask[i+st0] == 0 && data[i+st0] >= threshold)
	    bndry.push_back(Grid_Index(i0+1,i1,i2));
	  if (i1+1 < s1 && mask[i+st1] == 0 && data[i+st1] >= threshold)
	    bndry.push_back(Grid_Index(i0,i1+1,i2));
	  if (i2+1 < s2 && mask[i+1] == 0 && data[i+1] >= threshold)
	    bndry.push_back(Grid_Index(i0,i1,i2+1));
	}
    }

  return c;		  // Return number of regions found.
}

} // end of namespace Segmentation_Calculation
