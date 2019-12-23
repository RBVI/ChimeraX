// ----------------------------------------------------------------------------
// Routines for calculationg segmentations of volume data.
//
#ifndef MASK_HEADER_INCLUDED
#define MASK_HEADER_INCLUDED

#include <vector>		// use std::vector<>

namespace Segmentation_Calculation
{

typedef unsigned int Index;

Index largest_value(Index *mask, const int *mask_size);
void region_sizes(Index *mask, const int *mask_size, Index rmax, Index *rsize);
void region_grid_indices(Index *mask, const int *mask_size, int **grid_points);
void region_bounds(Index *mask, const int *mask_size, Index rmax,
		   int *bounds);
long region_point_count(Index *mask, const int *mask_size,
			const long *strides, Index rid);
long region_points(Index *mask, const int *mask_size,
		   const long *strides, Index rid, int *points);

class Contact
{
 public:
  Contact() { region1 = region2 = 0; ncontact = 0; data_max = -1e37; data_sum = 0; }
  Index region1, region2;
  unsigned int ncontact;
  float data_max;
  float data_sum;
};
typedef std::vector<Contact> Contacts;

void region_contacts(Index *mask, const int *mask_size, Contacts &contacts);


template <class T>
void interface_values(Index *mask, const int *mask_size, 
		      const T *data, Contacts &contacts);

template <class T>
void region_maxima(Index *mask, const int *mask_size, const T *data,
		   Index nmax, int *max_points, float *max_values);

} // end of namespace Segmentation_Calculation

#include "mask.cpp"		// Need template definitions

#endif
