// ----------------------------------------------------------------------------
// Routines for calculationg segmentations of volume data.
//
#ifndef REGION_MAP_HEADER_INCLUDED
#define REGION_MAP_HEADER_INCLUDED

#include <vector>		// use std::vector<>

namespace Segment_Map
{

typedef unsigned int Index;

Index largest_value(Index *region_map, const int64_t *region_map_size);
void region_sizes(Index *region_map, const int64_t *region_map_size, Index rmax, Index *rsize);
void region_grid_indices(Index *region_map, const int64_t *region_map_size, int **grid_points);
void region_bounds(Index *region_map, const int64_t *region_map_size, Index rmax,
		   int *bounds);
int64_t region_point_count(Index *region_map, const int64_t *region_map_size,
			   const int64_t *strides, Index rid);
int64_t region_points(Index *region_map, const int64_t *region_map_size,
		      const int64_t *strides, Index rid, int *points);

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

void region_contacts(Index *region_map, const int64_t *region_map_size, Contacts &contacts);


template <class T>
void interface_values(Index *region_map, const int64_t *region_map_size, 
		      const T *data, Contacts &contacts);

template <class T>
void region_maxima(Index *region_map, const int64_t *region_map_size, const T *data,
		   Index nmax, int *max_points, float *max_values);

} // end of namespace Segment_Map

#include "region_map.cpp"		// Need template definitions

#endif
