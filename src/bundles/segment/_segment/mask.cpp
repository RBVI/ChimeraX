// ----------------------------------------------------------------------------
//
#include <utility>		// use std::pair<>
#include <map>			// use std::map<>

#include "mask.h"		// use Index

namespace Segmentation_Calculation
{

typedef std::pair<Index, Index> Region_Pair;
typedef std::map<Region_Pair, Contact> Contact_Map;

// ----------------------------------------------------------------------------
//
Index largest_value(Index *mask, const int *mask_size)
{
  long s = mask_size[0] * mask_size[1] * mask_size[2];
  Index c = 0;
  for (long i = 0 ; i < s ; ++i)
    if (mask[i] > c)
      c = mask[i];
  return c;
}

// ----------------------------------------------------------------------------
//
void region_sizes(Index *mask, const int *mask_size, Index rmax, Index *rsize)
{
  for (Index r = 0 ; r <= rmax ; ++r)
    rsize[r] = 0;

  long s = mask_size[0] * mask_size[1] * mask_size[2];
  for (long i = 0 ; i < s ; ++i)
    rsize[mask[i]] += 1;
}

// ----------------------------------------------------------------------------
//
void region_bounds(Index *mask, const int *mask_size, Index rmax, int *bounds)
{
  int s0 = mask_size[0], s1 = mask_size[1], s2 = mask_size[2];
  int st0 = s1*s2, st1 = s2;

  int *b = bounds;
  for (Index r = 0 ; r <= rmax ; ++r, b += 7)
    {
      b[0] = mask_size[2];
      b[1] = mask_size[1];
      b[2] = mask_size[0];
      b[3] = b[4] = b[5] = b[6] = 0;
    }
  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1)
      for (int i2 = 0 ; i2 < s2 ; ++i2)
	{
	  long i = i0*st0 + i1*st1 + i2;
	  Index r = mask[i];
	  if (r <= rmax)
	    {
	      int *br = bounds + 7*r;
	      if (i2 < br[0]) br[0] = i2;
	      if (i1 < br[1]) br[1] = i1;
	      if (i0 < br[2]) br[2] = i0;
	      if (i2 > br[3]) br[3] = i2;
	      if (i1 > br[4]) br[4] = i1;
	      if (i0 > br[5]) br[5] = i0;
	      br[6] += 1;
	    }
	}
}

// ----------------------------------------------------------------------------
//
long region_point_count(Index *mask, const int *mask_size,
			const long *strides, Index rid)
{
  long count = 0;
  int s0 = mask_size[0], s1 = mask_size[1], s2 = mask_size[2];
  long st0 = strides[0], st1 = strides[1], st2 = strides[2];
  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1)
      for (int i2 = 0 ; i2 < s2 ; ++i2)
	{
	  long i = i0*st0 + i1*st1 + i2*st2;
	  if (mask[i] == rid)
	    count += 1;
	}
  return count;
}

// ----------------------------------------------------------------------------
//
long region_points(Index *mask, const int *mask_size,
		   const long *strides, Index rid, int *points)
{
  long count = 0;
  int s0 = mask_size[0], s1 = mask_size[1], s2 = mask_size[2];
  long st0 = strides[0], st1 = strides[1], st2 = strides[2];
  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1)
      for (int i2 = 0 ; i2 < s2 ; ++i2)
	{
	  long i = i0*st0 + i1*st1 + i2*st2;
	  if (mask[i] == rid)
	    {
	      int *p = points + 3*count;
	      p[0] = i2; p[1] = i1, p[2] = i0;
	      count += 1;
	    }
	}
  return count;
}

// ----------------------------------------------------------------------------
// Index arrays must already be allocated to the correct sizes.
// Index array pointers are incremented as a side effect.
//
void region_grid_indices(Index *mask, const int *mask_size, int **grid_points)
{
  int s0 = mask_size[0], s1 = mask_size[1], s2 = mask_size[2];
  int st0 = s1*s2, st1 = s2;
  int **gp = grid_points;

  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1)
      for (int i2 = 0 ; i2 < s2 ; ++i2)
	{
	  long i = i0*st0 + i1*st1 + i2;
	  Index r = mask[i];
	  int *p = gp[r];
	  if (p)
	    {
	      p[0] = i2; p[1] = i1; p[2] = i0;
	      gp[r] += 3;
	    }
	}
}

// ----------------------------------------------------------------------------
//
static Contact &add_contact(Index r1, Index r2, Contact_Map &rpmap)
{
  Region_Pair rp = (r1 < r2 ? Region_Pair(r1,r2) : Region_Pair(r2,r1));
  Contact &c = rpmap[rp];
  c.region1 = rp.first;
  c.region2 = rp.second;
  c.ncontact += 1;
  return c;
}

// ----------------------------------------------------------------------------
//
void region_contacts(Index *mask, const int *mask_size, Contacts &contacts)
{
  int s0 = mask_size[0], s1 = mask_size[1], s2 = mask_size[2];
  long st0 = s1*s2, st1 = s2;

  Contact_Map rpmap;
  Index r1, r2;
  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1)
      for (int i2 = 0 ; i2 < s2 ; ++i2)
	{
	  long i = i0*st0 + i1*st1 + i2;
	  r1 = mask[i];
	  if (r1 > 0)
	    {
	      if (i2+1 < s2 && (r2 = mask[i+1]) > 0 && r2 != r1)
		add_contact(r1, r2, rpmap);
	      if (i1+1 < s1 && (r2 = mask[i+st1]) > 0 && r2 != r1)
		add_contact(r1, r2, rpmap);
	      if (i0+1 < s0 && (r2 = mask[i+st0]) > 0 && r2 != r1)
		add_contact(r1, r2, rpmap);
	    }
	}

  for (Contact_Map::iterator c = rpmap.begin() ; c != rpmap.end() ; ++c)
    contacts.push_back(c->second);
}

// ----------------------------------------------------------------------------
//
static Contact &add_contact(Index r1, Index r2, float d1, float d2,
			    Contact_Map &rpmap)
{
  Region_Pair rp = (r1 < r2 ? Region_Pair(r1,r2) : Region_Pair(r2,r1));
  Contact &c = rpmap[rp];
  c.region1 = rp.first;
  c.region2 = rp.second;
  c.ncontact += 1;
  c.data_sum += d1 + d2;
  float d = (d1 > d2 ? d1 : d2);
  if (d > c.data_max)
    c.data_max = d;
  return c;
}

// ----------------------------------------------------------------------------
//
template <class T>
void interface_values(Index *mask, const int *mask_size, 
		      const T *data, Contacts &contacts)
{
  int s0 = mask_size[0], s1 = mask_size[1], s2 = mask_size[2];
  long st0 = s1*s2, st1 = s2;

  Contact_Map rpmap;
  Index r1, r2;
  float d;
  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1)
      for (int i2 = 0 ; i2 < s2 ; ++i2)
	{
	  long i = i0*st0 + i1*st1 + i2;
	  r1 = mask[i];
	  if (r1 > 0)
	    {
	      d = data[i];
	      if (i2+1 < s2 && (r2 = mask[i+1]) > 0 && r2 != r1)
		add_contact(r1, r2, d, (float)data[i+1], rpmap);
	      if (i1+1 < s1 && (r2 = mask[i+st1]) > 0 && r2 != r1)
		add_contact(r1, r2, d, (float)data[i+st1], rpmap);
	      if (i0+1 < s0 && (r2 = mask[i+st0]) > 0 && r2 != r1)
		add_contact(r1, r2, d, (float)data[i+st0], rpmap);
	    }
	}

  for (Contact_Map::iterator c = rpmap.begin() ; c != rpmap.end() ; ++c)
    contacts.push_back(c->second);
}

// ----------------------------------------------------------------------------
//
template <class T>
void region_maxima(Index *mask, const int *mask_size, const T *data,
		   Index nmax, int *max_points, float *max_values)
{
  int s0 = mask_size[0], s1 = mask_size[1], s2 = mask_size[2];
  int st0 = s1*s2, st1 = s2;

  for (Index p = 0 ; p < nmax ; ++p)
    max_values[p] = -1e37;

  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1)
      for (int i2 = 0 ; i2 < s2 ; ++i2)
	{
	  long i = i0*st0 + i1*st1 + i2;
	  Index r1 = mask[i];
	  if (r1 > 0 && r1 <= nmax)
	    {
	      float d = data[i];
	      if (d > max_values[r1-1])
		{
		  max_values[r1-1] = d;
		  int *p = &max_points[3*(r1-1)];
		  p[0] = i2; p[1] = i1; p[2] = i0;
		}
	    }
	}
}

} // end of namespace Segmentation_Calculation
