/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
// Find pairs of close points given two sets of points and a distance.
//
#include <Python.h>			// use PyTuple*(), ...
#include <cstdint>			// use std::int64_t
#include <iostream>			// use std:cerr for debugging
#include <map>				// use map
#include <vector>			// use vector

#include <math.h>			// use floor()

#include <arrays/pythonarray.h>		// use float_2d_array_values()
#include <arrays/rcarray.h>		// use FArray
#include "closepoints.h"

using std::vector;
using std::map;
using std::pair;

typedef std::int64_t Index;
typedef vector<Index> Index_List;

typedef enum {CP_ALL_PAIRS, CP_BOX_ALL_PAIRS, CP_BINS, CP_BOX_BINS, CP_BOXES}
   Close_Points_Method;

// ---------------------------------------------------------------------------
// Find close points between two sets of points.  The returned lists i1 and i2
// hold the indices from sets 1 and 2 of points that are part of close pairs.
// The returned nearest list gives the indices of set 2 points that are nearest
// to the set 1 points and within range d.  Its elements correspond to those in returned
// array i1. The nearest pointer can be NULL if that information is not
// needed.  If the scales argument is not NULL then the distance from set 2
// point i is scaled by 1/scales[i] when computing the nearest point.
//
void find_close_points(Close_Points_Method m,
		       const float *xyz1, Index n1,
		       const float *xyz2, Index n2,
		       float d, float *scales,
		       Index_List *i1, Index_List *i2,
		       Index_List *nearest1 = NULL);

// ---------------------------------------------------------------------------
// Find close points between multiple sets of transformed points.
//
class Transformed_Points
{
 public:
  const float *xyz;
  Index n;
  float rotation[3][3];		// Rotation matrix
  float translation[3];		// Translation vector (after rotation).
};
void find_close_points(Close_Points_Method m,
		       const vector<Transformed_Points> &p1,
		       const vector<Transformed_Points> &p2,
		       float distance,
		       vector<Index_List> *i1,
		       vector<Index_List> *i2);

// ----------------------------------------------------------------------------
//
class Index3
{
 public:
  Index3(int x, int y, int z) { this->x = x; this->y = y; this->z = z; }
  int x, y, z;
  bool operator<(const Index3 &i3) const
    { return (x < i3.x || (x == i3.x && (y < i3.y || (y == i3.y && z < i3.z)))); }
  bool operator==(const Index3 &i3) const
    { return (x == i3.x && y == i3.y && z == i3.z); }
};

// ----------------------------------------------------------------------------
//
typedef map<Index3, Index_List> Bin_Map;

// ----------------------------------------------------------------------------
//
class Index_Set
{
public:
  Index_Set(Index_List *ilist, Index n) : have_index(n, 0)
    {
      this->ilist = ilist;
      Index_List::size_type m = ilist->size();
      for (Index_List::size_type k = 0 ; k < m ; ++k)
	{
	  Index i = (*ilist)[k];
	  if (i < n)
	    have_index[i] = 1;
	}
    }
  void add_index(Index i)
    { if (have_index[i] == 0) { have_index[i] = 1; ilist->push_back(i); } }
  bool in_set(Index i) const
    { return have_index[i]; }
  Index size() const
    { return ilist->size(); }
private:
  Index_List *ilist;
  // TODO: Maybe don't create have_index array unless it is used.
  vector<int> have_index;
};


// ----------------------------------------------------------------------------
//
class Nearest_Points
{
public:
  Nearest_Points(Index n)
  {
    this->closest = new Index[n];
    this->d2 = new float[n];
    for (Index k = 0 ; k < n ; ++k)
      {
	d2[k] = NO_DISTANCE;
	closest[k] = -1;
      }
  }
  virtual ~Nearest_Points()
  {
    delete [] closest;
    closest = NULL;
    delete [] d2;
    d2 = NULL;
  }
  virtual void close_pair(Index i, Index j, float d2_ij)
  {
    float d2min = d2[i];
    if (d2min == NO_DISTANCE || d2_ij < d2min)
      {
	closest[i] = j;
	d2[i] = d2_ij;
      }
  }
  void nearest_list(const Index_List &i1, Index_List &nearest1)
  {
    Index m = i1.size();
    nearest1.resize(m,0);
    for (Index k = 0 ; k < m ; ++k)
      nearest1[k] = closest[i1[k]];
  }

protected:
  Index *closest;
  float *d2;
  static const float NO_DISTANCE;
};
const float Nearest_Points::NO_DISTANCE = -1.0;

// ----------------------------------------------------------------------------
// Scale each distance to the first point by a scale factor when determining
// which neighbor point is nearest.
//
class Nearest_Scaled_Points : public Nearest_Points
{
public:
  Nearest_Scaled_Points(Index n, float *scales) : Nearest_Points(n), scales(scales) {}
  virtual void close_pair(Index i, Index j, float d2_ij)
  {
    float d2min = d2[i];
    if (d2min == NO_DISTANCE)
      {
	closest[i] = j;
	d2[i] = d2_ij;
      }
    else 
      {
	float sc = scales[closest[i]], sj = scales[j];
	if (sc*sc*d2_ij < sj*sj*d2min)
	  {
	    closest[i] = j;
	    d2[i] = d2_ij;
	  }
      }
  }

private:
  float *scales;
};

// ----------------------------------------------------------------------------
//
class Box
{
 public:
  float xyz_min[3], xyz_max[3];

  float size(int axis) const;
  int long_axis() const;
  float volume() const;
  void extend(float pad, Box *result) const;
  bool intersect(const Box &box2, Box *result) const;
  bool empty() const;
  void split(int axis, Box *result1, Box *result2) const;
};

// ----------------------------------------------------------------------------
//
class Point_List
{
public:
  const float *xyz;
  Point_List(const float *xyz, Index n)
    {
      this->xyz = xyz;
      this->nxyz = n;
      this->ilist = NULL;
      this->nilist = 0;
      this->delete_ilist = false;
      this->bbox_valid = false;
    }
  Point_List(const float *xyz, Index nxyz, Index *ilist, Index ni,
	     bool delete_ilist = false)
    {
      this->xyz = xyz;
      this->nxyz = nxyz;
      this->ilist = ilist;
      this->nilist = ni;
      this->delete_ilist = delete_ilist;
      this->bbox_valid = false;
    }
  Point_List(const Point_List &p)
    {
      this->xyz = p.xyz;
      this->nxyz = p.nxyz;
      if (p.ilist)
	{
	  Index ni = p.nilist;
	  this->ilist = new Index[ni];
	  for (Index k = 0 ; k < ni ; ++k)
	    ilist[k] = p.ilist[k];
	  this->delete_ilist = true;
	}
      else
	{
	  this->ilist = NULL;
	  this->delete_ilist = false;
	}
      this->nilist = p.nilist;
      this->bbox_valid = p.bbox_valid;
      if (p.bbox_valid)
	this->bbox = p.bbox;
    }

  virtual ~Point_List()
    {
      if (delete_ilist)
	{
	  delete [] ilist;
	  ilist = NULL;
	}
    }
  Index size() const
    { return (ilist ? nilist : nxyz); }
  Index index_range() const
    { return nxyz; }
  Index index(Index k) const
    { return (ilist ? ilist[k] : k); }
  // TODO: Use separate classes for ilist vs no ilist to improve
  //       indexing inlining performance.
  void restrict_to_box(const Box &b);
  bool have_bounding_box() const { return bbox_valid; }
  const Box &bounding_box() const;
private:
  Index nxyz;
  Index *ilist;
  Index nilist;
  bool delete_ilist;
  mutable Box bbox;
  mutable bool bbox_valid;
};

// ----------------------------------------------------------------------------
//
class BBox_Cache
{
public:
  virtual ~BBox_Cache();
  void bounding_box(const Transformed_Points &tp, Box *box, Point_List **pl,
		    bool require_minimal_box = false);
private:
  typedef map<const Transformed_Points *, Box> BTable;
  BTable btable;
  typedef map<const Transformed_Points *, Point_List *> PTable;
  PTable ptable;
  typedef map<pair<const float*,int>, const float *> TBTable;
  TBTable tbtable;

  Point_List *point_list(const Transformed_Points &tp);
};

// ----------------------------------------------------------------------------
//
class Contacts
{
 public:
  Contacts() { finished = false; }
  virtual ~Contacts() {}
  virtual void add_contact(Index /*i1*/, Index /*i2*/) { finished = true; }
  virtual void all_in_contact() { finished = true; }
  bool finished;
};

// ----------------------------------------------------------------------------
//
static void find_close_points(Close_Points_Method m,
			      const Point_List &p1, const Point_List &p2,
			      float d, Index_Set &i1, Index_Set &i2,
			      Nearest_Points *np1);
static void find_close_points_all_pairs(const float *xyz1, Index n1,
					const float *xyz2, Index n2,
					float d,
					Index_Set &is1, Index_Set &is2,
					Nearest_Points *np1);
static void find_close_points_all_pairs(const float *xyz1, const Index_List &i1,
					const float *xyz2, const Index_List &i2,
					float d,
					Index_Set &is1, Index_Set &is2,
					Nearest_Points *np1);
static void find_close_points_all_pairs(const Point_List &p1,
					const Point_List &p2,
					float d,
					Index_Set &is1, Index_Set &is2,
					Nearest_Points *np1);
static void find_close_points_bins(const Point_List &p1, const Point_List &p2,
				   float d, Index_Set &is1, Index_Set &is2,
				   Nearest_Points *np1);
static void find_close_points_boxes(const Point_List &p1, const Point_List &p2,
				    float d, Index_Set &is1, Index_Set &is2,
				    Nearest_Points *np1);
static void find_close_points_subboxes(const Point_List &p1, Box *box1,
				       const Point_List &p2, Box *box2,
				       float d, Index_Set &is1, Index_Set &is2,
				       Nearest_Points *np1);
static void split_point_list(const Point_List &p, int axis,
			     Point_List **p1, Point_List **p2);
static void split_point_list(const Point_List &p, int axis,
			     Index **i1, Index *ni1, Index **i2, Index *ni2);
static float maximum_separation_squared(const Box &box1, const Box &box2);
static float minimum_separation_squared(const Box &box1, const Box &box2);
static void add_points_to_set(const Point_List &p, Index_Set &is);
static Index points_in_box(const Point_List &p, const Box &box, Index **i, Index *ni);
static void reduce_to_box_intersection(Point_List &p1, Point_List &p2,
				       float d, float volume_threshold);
static bool reduce_to_box_intersection(Point_List &p, const Box &box,
			       float d, float volume_threshold, bool &change);
static void transform_points(const float *xyz, Index n,
			     const float rotation[3][3],
			     const float translation[3], float *txyz);
static void inverse_transform_points(const float *xyz, Index n,
				     const float rotation[3][3],
				     const float translation[3],
				     float *itxyz);
static void box_corners(const Box &box, float *corners);
static void transformed_points_bounding_box(const float *xyz, Index n,
					    const float rotation[3][3],
					    const float translation[3],
					    Box *box);
static bool boxes_are_close(const Box &box1, const Box &box2, float distance);

// ----------------------------------------------------------------------------
//
void find_close_points(Close_Points_Method m,
		       const float *xyz1, Index n1,
		       const float *xyz2, Index n2,
		       float d, float *scales,
		       Index_List *i1, Index_List *i2, Index_List *nearest1)
{
  Point_List p1(xyz1, n1), p2(xyz2, n2);
  Index_Set is1(i1, n1), is2(i2, n2);
  Nearest_Points *np1 = NULL;
  if (nearest1)
    np1 = (scales ? new Nearest_Scaled_Points(n1,scales) : new Nearest_Points(n1));
  find_close_points(m, p1, p2, d, is1, is2, np1);
  if (np1)
    {
      np1->nearest_list(*i1, *nearest1);
      delete np1;
    }
}

// ----------------------------------------------------------------------------
//
static void find_close_points(Close_Points_Method m,
			      const Point_List &p1, const Point_List &p2,
			      float d, Index_Set &is1, Index_Set &is2,
			      Nearest_Points *np1)
{
  const float volume_threshold = .5;
  switch (m)
    {
    case CP_ALL_PAIRS:
      {
	find_close_points_all_pairs(p1, p2, d, is1, is2, np1);
	break;
      }
    case CP_BOX_ALL_PAIRS:
      {
	// Copy point lists so they can be modified.
	Point_List p1c(p1), p2c(p2);
	reduce_to_box_intersection(p1c, p2c, d, volume_threshold);
	find_close_points_all_pairs(p1c, p2c, d, is1, is2, np1);
	break;
      }
    case CP_BINS:
      {
	find_close_points_bins(p1, p2, d, is1, is2, np1);
	break;
      }
    case CP_BOX_BINS:
      {
	// Copy point lists so they can be modified.
	Point_List p1c(p1), p2c(p2);
	reduce_to_box_intersection(p1c, p2c, d, volume_threshold);
	find_close_points_bins(p1c, p2c, d, is1, is2, np1);
	break;
      }
    case CP_BOXES:
      {
	find_close_points_boxes(p1, p2, d, is1, is2, np1);
	break;
      }
    default:
      std::cerr << "Warning: find_close_points() called with non-existent method "
		<< m << std::endl;
    }
}

// ----------------------------------------------------------------------------
//
static void find_close_points_all_pairs(const float *xyz1, Index n1,
					const float *xyz2, Index n2,
					float d,
					Index_Set &is1, Index_Set &is2,
					Nearest_Points *np1)
{
  float d2_limit = d * d;
  Index s1 = 3*n1, s2 = 3*n2;
  for (Index j1 = 0 ; j1 < s1 ; j1 += 3)
    {
      float x1 = xyz1[j1], y1 = xyz1[j1+1], z1 = xyz1[j1+2];
      for (Index j2 = 0 ; j2 < s2 ; j2 += 3)
	{
	  float dx = xyz2[j2] - x1;
	  float dy = xyz2[j2+1] - y1;
	  float dz = xyz2[j2+2] - z1;
	  float d2 = dx*dx + dy*dy + dz*dz;
	  if (d2 <= d2_limit)
	    {
	      is1.add_index(j1/3);
	      is2.add_index(j2/3);
	      if (np1)
		np1->close_pair(j1/3, j2/3, d2);
	    }
	}
    }
}

// ----------------------------------------------------------------------------
//
static void find_close_points_all_pairs(const float *xyz1, const Index_List &i1,
					const float *xyz2, const Index_List &i2,
					float d,
					Index_Set &is1, Index_Set &is2,
					Nearest_Points *np1)
{
  Index n1 = i1.size(), n2 = i2.size();
  float d2_limit = d*d;
  for (Index k1 = 0 ; k1 < n1 ; ++k1)
    {
      Index i1ind = i1[k1];
      Index i1ind3 = 3 * i1ind;
      float x1 = xyz1[i1ind3], y1 = xyz1[i1ind3+1], z1 = xyz1[i1ind3+2];
      for (Index k2 = 0 ; k2 < n2 ; ++k2)
	{
	  Index i2ind = i2[k2];
	  Index i2ind3 = 3 * i2ind;
	  float dx = xyz2[i2ind3] - x1;
	  float dy = xyz2[i2ind3+1] - y1;
	  float dz = xyz2[i2ind3+2] - z1;
	  float d2 = dx*dx + dy*dy + dz*dz;
	  if (d2 <= d2_limit)
	    {
	      is1.add_index(i1ind);
	      is2.add_index(i2ind);
	      if (np1)
		np1->close_pair(i1ind, i2ind, d2);
	    }
	}
    }
}

// ----------------------------------------------------------------------------
//
static void find_close_points_all_pairs(const Point_List &p1,
					const Point_List &p2,
					float d,
					Index_Set &is1, Index_Set &is2,
					Nearest_Points *np1)
{
  Index n1 = p1.size(), n2 = p2.size();
  if (p1.index_range() == n1 && p2.index_range() == n2)
    {
      // Optimization for non-sublist case.
      find_close_points_all_pairs(p1.xyz, n1, p2.xyz, n2, d, is1, is2, np1);
      return;
    }
  float d2_limit = d*d;
  for (int64_t k1 = 0 ; k1 < n1 ; ++k1)
    {
      Index i1ind = p1.index(k1);
      Index i1ind3 = 3 * i1ind;
      float x1 = p1.xyz[i1ind3], y1 = p1.xyz[i1ind3+1], z1 = p1.xyz[i1ind3+2];
      for (int64_t k2 = 0 ; k2 < n2 ; ++k2)
	{
	  Index i2ind = p2.index(k2);
	  Index i2ind3 = 3 * i2ind;
	  float dx = p2.xyz[i2ind3] - x1;
	  float dy = p2.xyz[i2ind3+1] - y1;
	  float dz = p2.xyz[i2ind3+2] - z1;
	  float d2 = dx*dx + dy*dy + dz*dz;
	  if (d2 <= d2_limit)
	    {
	      is1.add_index(i1ind);
	      is2.add_index(i2ind);
	      if (np1)
		np1->close_pair(i1ind, i2ind, d2);
	    }
	}
    }
}

// ----------------------------------------------------------------------------
//
static void bin_points(const Point_List &p, float d, Bin_Map &bins)
{
  Index n = p.size();
  for (int64_t k = 0 ; k < n ; ++k)
    {
      Index i = p.index(k);
      Index i3 = 3*i;
      float x = p.xyz[i3], y = p.xyz[i3+1], z = p.xyz[i3+2];
      Index3 b(static_cast<int>(floor(x/d)),
	       static_cast<int>(floor(y/d)),
	       static_cast<int>(floor(z/d)));
      bins[b].push_back(i);
    }
}

// ----------------------------------------------------------------------------
//
static void find_close_points_bins(const Point_List &p1, const Point_List &p2,
				   float d, Index_Set &is1, Index_Set &is2,
				   Nearest_Points *np1)
{
  Bin_Map b1, b2;
  bin_points(p1, d, b1);
  bin_points(p2, d, b2);

  for (Bin_Map::iterator b1i = b1.begin() ; b1i != b1.end() ; ++b1i)
    {
      const Index3 &b = b1i->first;
      Index_List &b1ind = b1i->second;
      for (int xoffset = -1 ; xoffset <= 1 ; ++xoffset)
	for (int yoffset = -1 ; yoffset <= 1 ; ++yoffset)
	  for (int zoffset = -1 ; zoffset <= 1 ; ++zoffset)
	    {
	      Index3 bo(b.x+xoffset, b.y+yoffset, b.z+zoffset);
	      Bin_Map::iterator b2i = b2.find(bo);
	      if (b2i != b2.end())
		find_close_points_all_pairs(p1.xyz, b1ind, p2.xyz, b2i->second,
					    d, is1, is2, np1);
	    }
    }
}

// ----------------------------------------------------------------------------
//
static void find_close_points_boxes(const Point_List &p1, const Point_List &p2,
				    float d, Index_Set &is1, Index_Set &is2,
				    Nearest_Points *np1)
{
  const float volume_threshold = .5;
  const Index n_cutoff = 20;

  if (p1.size() < n_cutoff || p2.size() < n_cutoff)
    {
      find_close_points_all_pairs(p1, p2, d, is1, is2, np1);
      return;
    }

  const Box &box1 = p1.bounding_box(), &box2 = p2.bounding_box();

  float d2 = d*d;
  if (minimum_separation_squared(box1, box2) >= d2)
    // Optimization for d large compared to size of point sets.
    return;

  if (maximum_separation_squared(box1, box2) <= d2)
    {
      // Optimization for d large compared to size of point sets.
      // All points of both sets are in contact.
      if (np1 == NULL)
	{
	  add_points_to_set(p1, is1);
	  add_points_to_set(p2, is2);
	  return;
	}
      else if (box2.size(0) == 0 && box2.size(1) == 0 && box2.size(2) == 0 && p2.size() > 0)
	{
	  // If lots of identical points are in list 2 avoid recursing forever.
	  Index i2 = p2.index(0);
	  Point_List p20(p2.xyz, p2.index_range(), &i2, 1);
	  find_close_points_all_pairs(p1, p20, d, is1, is2, np1);
	  return;
	}
    }

  Box ebox1, ebox2;
  box1.extend(d, &ebox1);
  box2.extend(d, &ebox2);
  Box ibox1, ibox2;
  if (!box1.intersect(ebox2, &ibox1) || !box2.intersect(ebox1, &ibox2))
    return;	// Distance between boxes is greater than d.

  bool f1 = (ibox1.volume() < volume_threshold * box1.volume());
  bool f2 = (ibox2.volume() < volume_threshold * box2.volume());
  if (f1 || f2)
    {
      find_close_points_subboxes(p1, (f1 ? &ibox1 : NULL),
				 p2, (f2 ? &ibox2 : NULL),
				 d, is1, is2, np1);
      return;
    }

  //
  // Split boxes.
  //
  Point_List *p1a, *p1b, *p2a, *p2b;
  int a1 = box1.long_axis(), a2 = box2.long_axis();
  split_point_list(p1, a1, &p1a, &p1b);
  split_point_list(p2, a2, &p2a, &p2b);
  find_close_points_boxes(*p1a, *p2a, d, is1, is2, np1);
  find_close_points_boxes(*p1a, *p2b, d, is1, is2, np1);
  find_close_points_boxes(*p1b, *p2a, d, is1, is2, np1);
  find_close_points_boxes(*p1b, *p2b, d, is1, is2, np1);
  delete p1a;
  delete p1b;
  delete p2a;
  delete p2b;
}

// ----------------------------------------------------------------------------
//
static void find_close_points_subboxes(const Point_List &p1, Box *box1,
				       const Point_List &p2, Box *box2,
				       float d, Index_Set &is1, Index_Set &is2,
				       Nearest_Points *np1)
{
  if (box1 && box2)
    {
      Index *ib1, nib1, *ib2, nib2;
      if (points_in_box(p1, *box1, &ib1, &nib1) == 0)
	return;
      if (points_in_box(p2, *box2, &ib2, &nib2) == 0)
	{
	  delete [] ib1;
	  return;
	}
      Point_List p1f(p1.xyz, p1.index_range(), ib1, nib1, true);
      Point_List p2f(p2.xyz, p2.index_range(), ib2, nib2, true);
      find_close_points_boxes(p1f, p2f, d, is1, is2, np1);
    }
  else if (box1)
    {
      Index *ib1, nib1;
      if (points_in_box(p1, *box1, &ib1, &nib1) == 0)
	return;
      Point_List p1f(p1.xyz, p1.index_range(), ib1, nib1, true);
      find_close_points_boxes(p1f, p2, d, is1, is2, np1);
    }
  else if (box2)
    {
      Index *ib2, nib2;
      if (points_in_box(p2, *box2, &ib2, &nib2) == 0)
	return;
      Point_List p2f(p2.xyz, p2.index_range(), ib2, nib2, true);
      find_close_points_boxes(p1, p2f, d, is1, is2, np1);
    }
}

// ----------------------------------------------------------------------------
//
static void split_point_list(const Point_List &p, int axis,
			     Point_List **p1, Point_List **p2)
{
  // TODO: Should split box in one loop comparing only one axis for speed.
  Index *i1, ni1, *i2, ni2;
  split_point_list(p, axis, &i1, &ni1, &i2, &ni2);
  *p1 = new Point_List(p.xyz, p.index_range(), i1, ni1, true);
  *p2 = new Point_List(p.xyz, p.index_range(), i2, ni2, true);
}

// ----------------------------------------------------------------------------
//
static void split_point_list(const Point_List &p, int axis,
			     Index **i1, Index *ni1, Index **i2, Index *ni2)
{
  const Box &box = p.bounding_box();
  float amid = .5 * (box.xyz_min[axis] + box.xyz_max[axis]);
  Index n = p.size();
  Index *il = new Index[n];
  Index li = 0, gi = n-1;
  const float *axyz = p.xyz + axis;
  for (int64_t k = 0 ; k < n ; ++k)
    {
      Index i = p.index(k);
      Index i3 = 3*i;
      float ap = axyz[i3];
      if (ap >= amid)
	il[li++] = i;
      else
	il[gi--] = i;
    }

  Index *ic1 = new Index[li];
  Index *ic2 = new Index[n-1-gi];
  for (int64_t k = 0 ; k < li ; ++k)
    ic1[k] = il[k];
  for (int64_t k = n-1 ; k > gi ; --k)
    ic2[n-1-k] = il[k];
  delete [] il;

  *ni1 = li;
  *ni2 = n - 1 - gi;
  *i1 = ic1;
  *i2 = ic2;
}

// ----------------------------------------------------------------------------
//
static float maximum_separation_squared(const Box &box1, const Box &box2)
{
  float sum = 0;
  for (int a = 0 ; a < 3 ; ++a)
    {
      float q1min = box1.xyz_min[a], q2min = box2.xyz_min[a];
      float qmin = (q1min < q2min ? q1min : q2min);
      float q1max = box1.xyz_max[a], q2max = box2.xyz_max[a];
      float qmax = (q1max > q2max ? q1max : q2max);
      float dq = qmax - qmin;
      sum += dq*dq;
    }
  return sum;
}

// ----------------------------------------------------------------------------
//
static float minimum_separation_squared(const Box &box1, const Box &box2)
{
  float sum = 0;
  for (int a = 0 ; a < 3 ; ++a)
    {
      float q1max = box1.xyz_max[a], q2min = box2.xyz_min[a];
      if (q2min > q1max)
	{
	  float dq = q2min - q1max;
	  sum += dq * dq;
	  continue;
	}
      float q1min = box1.xyz_min[a], q2max = box2.xyz_max[a];
      if (q1min > q2max)
	{
	  float dq = q1min - q2max;
	  sum += dq * dq;
	}
    }
  return sum;
}

// ----------------------------------------------------------------------------
//
static void add_points_to_set(const Point_List &p, Index_Set &is)
{
  Index n = p.size();
  for (int64_t k = 0 ; k < n ; ++k)
    is.add_index(p.index(k));
}

// ----------------------------------------------------------------------------
//
float Box::size(int axis) const
{
  float s = xyz_max[axis] - xyz_min[axis];
  return (s >= 0 ? s : 0);
}

// ----------------------------------------------------------------------------
//
int Box::long_axis() const
{
  float s0 = size(0), s1 = size(1), s2 = size(2);
  return (s0 > s1 ? (s0 > s2 ? 0 : 2) : (s1 > s2 ? 1 : 2));
}

// ----------------------------------------------------------------------------
//
float Box::volume() const
{
  float s0 = size(0), s1 = size(1), s2 = size(2);
  return s0 * s1 * s2;
}

// ----------------------------------------------------------------------------
//
void Box::extend(float pad, Box *result) const
{
  *result = *this;
  for (int a = 0 ; a < 3 ; ++a)
    {
      result->xyz_min[a] -= pad;
      result->xyz_max[a] += pad;
    }
}

// ----------------------------------------------------------------------------
//
bool Box::intersect(const Box &box2, Box *result) const
{
  for (int a = 0 ; a < 3 ; ++a)
    {
      float x1 = xyz_min[a], x2 = box2.xyz_min[a];
      result->xyz_min[a] = (x1 > x2 ? x1 : x2);
      x1 = xyz_max[a], x2 = box2.xyz_max[a];
      result->xyz_max[a] = (x1 < x2 ? x1 : x2);
    }
  return !empty();
}

// ----------------------------------------------------------------------------
//
bool Box::empty() const
{
  return (xyz_min[0] > xyz_max[0] ||
	  xyz_min[1] > xyz_max[1] ||
	  xyz_min[2] > xyz_max[2]);
}

// ----------------------------------------------------------------------------
//
void Box::split(int a, Box *s1, Box *s2) const
{
  float mid = .5 * (xyz_min[a] + xyz_max[a]);
  *s1 = *this;
  s1->xyz_max[a] = mid;
  *s2 = *this;
  s2->xyz_min[a] = mid;
}

// ----------------------------------------------------------------------------
//
const Box &Point_List::bounding_box() const
{
  if (bbox_valid)
    return bbox;

  float xmin, xmax, ymin, ymax, zmin, zmax;

  Index n = size();
  if (n == 0)
    {
      xmin = ymin = zmin = 0;
      xmax = ymax = zmax = -1;
    }
  else
    {
      Index i3 = 3*index(0);
      xmin = xyz[i3]; ymin = xyz[i3+1]; zmin = xyz[i3+2];
      xmax = xmin; ymax = ymin; zmax = zmin;
    }

  for (int64_t k = 1 ; k < n ; ++k)
    {
      Index i3 = 3*index(k);
      float x = xyz[i3], y = xyz[i3+1], z = xyz[i3+2];
      if (x < xmin) xmin = x;
      else if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      else if (y > ymax) ymax = y;
      if (z < zmin) zmin = z;
      else if (z > zmax) zmax = z;
    }
  bbox.xyz_min[0] = xmin;  bbox.xyz_min[1] = ymin;  bbox.xyz_min[2] = zmin;
  bbox.xyz_max[0] = xmax;  bbox.xyz_max[1] = ymax;  bbox.xyz_max[2] = zmax;
  bbox_valid = true;

  return bbox;
}

// ----------------------------------------------------------------------------
//
static Index points_in_box(const Point_List &p, const Box &box,
			 Index **ilist, Index *nilist)
{
  float x, y, z;
  float xmin = box.xyz_min[0], ymin = box.xyz_min[1], zmin = box.xyz_min[2];
  float xmax = box.xyz_max[0], ymax = box.xyz_max[1], zmax = box.xyz_max[2];
  Index n = p.size();
  Index *il = new Index[n];
  Index ni = 0;
  for (int64_t k = 0 ; k < n ; ++k)
    {
      Index i = p.index(k);
      Index i3 = 3*i;
      x = p.xyz[i3];
      if (x >= xmin && x <= xmax)
	{
	  y = p.xyz[i3+1];
	  if (y >= ymin && y <= ymax)
	    {
	      z = p.xyz[i3+2];
	      if (z >= zmin && z <= zmax)
		il[ni++] = i;
	    }
	}
    }

  Index *iresize = (ni > 0 ? new Index[ni] : NULL);
  for (int64_t k = 0 ; k < ni ; ++k)
    iresize[k] = il[k];
  delete [] il;

  *ilist = iresize;
  *nilist = ni;

  return ni;
}

// ----------------------------------------------------------------------------
//
static void reduce_to_box_intersection(Point_List &p1, Point_List &p2,
				       float d, float volume_threshold)
{
  if (p1.size() == 0 || p2.size() == 0)
    return;

  bool change1, change2;
  reduce_to_box_intersection(p1, p2.bounding_box(), d, volume_threshold, change1);
  while (p1.size() > 0 &&
	 reduce_to_box_intersection(p2, p1.bounding_box(), d, volume_threshold, change2) &&
	 p2.size() > 0 &&
	 reduce_to_box_intersection(p1, p2.bounding_box(), d, volume_threshold, change1))
    if (!(change1 || change2))
      break;
}

// ----------------------------------------------------------------------------
//
static bool reduce_to_box_intersection(Point_List &p, const Box &box,
			       float d, float volume_threshold, bool &change)
{
  Box ebox;
  box.extend(d, &ebox);
  Box pbox = p.bounding_box();
  Box ibox;
  pbox.intersect(ebox, &ibox);

  change = false;
  if (ibox.volume() < volume_threshold * pbox.volume())
    {
      Index pre = p.size();
      p.restrict_to_box(ibox);
      if (pre > p.size())
        change = true;
      return true;
    }
  return false;
}

// ----------------------------------------------------------------------------
//
void Point_List::restrict_to_box(const Box &box)
{
  Index *i, ni;
  points_in_box(*this, box, &i, &ni);
  if (ilist && delete_ilist)
    delete [] ilist;

  this->ilist = i;
  this->nilist = ni;
  this->delete_ilist = true;
  this->bbox_valid = false;
}

// ----------------------------------------------------------------------------
// Find contacts between one group of point sets, and another group of
// point sets.
//
void find_close_points(Close_Points_Method m,
		       const vector<Transformed_Points> &p1,
		       const vector<Transformed_Points> &p2,
		       float distance,
		       vector<Index_List> *i1, vector<Index_List> *i2)
{
  typedef map<int, Index_Set *> ISTable;
  ISTable is1, is2;
  BBox_Cache bbox_cache;
  Index n1 = p1.size(), n2 = p2.size();
  for (Index k1 = 0 ; k1 < n1 ; ++k1)
    {
      const Transformed_Points &tp1 = p1[k1];
      Box bbox1;
      Point_List *pl1;
      bbox_cache.bounding_box(tp1, &bbox1, &pl1);
      for (Index k2 = 0 ; k2 < n2 ; ++k2)
	{
	  const Transformed_Points &tp2 = p2[k2];
	  Box bbox2;
	  Point_List *pl2;
	  bbox_cache.bounding_box(tp2, &bbox2, &pl2);
	  if (boxes_are_close(bbox1, bbox2, distance))
	    {
	      if (pl1 == NULL || pl2 == NULL)
		{
		  // Recompute tighter bounding boxes from actual coordinates
		  if (pl1 == NULL)
		    bbox_cache.bounding_box(tp1, &bbox1, &pl1, true);
		  if (pl2 == NULL)
		    bbox_cache.bounding_box(tp2, &bbox2, &pl2, true);
		  if (! boxes_are_close(bbox1, bbox2, distance))
		    continue;
		}

	      if (static_cast<int>((*i1)[k1].size()) == tp1.n && 
		  static_cast<int>((*i2)[k2].size()) == tp2.n)
		// Optimization for d >> size of individual point lists.
		continue;

	      // Make copy of point lists because box filtering methods
	      // reduce point list to sublist.
	      Point_List pl1c(*pl1);
	      Point_List pl2c(*pl2);
	      if (is1.find(k1) == is1.end())
		is1[k1] = new Index_Set(&(*i1)[k1], pl1c.index_range());
	      if (is2.find(k2) == is2.end())
		is2[k2] = new Index_Set(&(*i2)[k2], pl2c.index_range());
	      find_close_points(m, pl1c, pl2c, distance, *is1[k1], *is2[k2],
				NULL);
	    }
	}
    }

  // Delete all index sets cached in is1 and is2 tables.
  for (ISTable::iterator ti = is1.begin() ; ti != is1.end() ; ++ti)
    delete (*ti).second;
  for (ISTable::iterator ti = is2.begin() ; ti != is2.end() ; ++ti)
    delete (*ti).second;
}

// ----------------------------------------------------------------------------
//
BBox_Cache::~BBox_Cache()
{
  for (PTable::iterator i = ptable.begin() ; i != ptable.end() ; ++i)
    {
      Point_List *p = i->second;
      if (p)
	{
	  delete [] p->xyz;
	  delete p;
	}
    }
  ptable.clear();

  for (TBTable::iterator i = tbtable.begin() ; i != tbtable.end() ; ++i)
    {
      const float *corners = i->second;
      delete [] corners;
    }
  tbtable.clear();
}

// ----------------------------------------------------------------------------
//
void BBox_Cache::bounding_box(const Transformed_Points &tp,
			      Box *box, Point_List **pl,
			      bool require_minimal_box)
{
  if (require_minimal_box)
    {
      Point_List *p = point_list(tp);
      *pl = p;
      *box = p->bounding_box();
      return;
    }

  // See if a box for this transformed point set is already cached.
  BTable::iterator i = btable.find(&tp);
  if (i != btable.end())
    {
      *box = i->second;
      PTable::iterator j = ptable.find(&tp);
      *pl = (j == ptable.end() ? NULL : j->second);
      return;
    }

  // Look for box for same coordinates, only transformed.
  pair<const float*,int> k(tp.xyz, tp.n);
  TBTable::iterator j = tbtable.find(k);
  if (j != tbtable.end())
    {
      const float *corners = j->second;
      transformed_points_bounding_box(corners, 8, tp.rotation, tp.translation,
				      box);
      btable[&tp] = *box;
      *pl = NULL;
      return;
    }

  Point_List *p = point_list(tp);
  *pl = p;
  *box = p->bounding_box();
}

// ----------------------------------------------------------------------------
//
Point_List *BBox_Cache::point_list(const Transformed_Points &tp)
{
  Point_List *p;
  PTable::iterator i = ptable.find(&tp);
  if (i == ptable.end())
    {
      float *txyz = new float[3*tp.n];
      transform_points(tp.xyz, tp.n, tp.rotation, tp.translation, txyz);
      p = new Point_List(txyz, tp.n);
      ptable[&tp] = p;
      btable[&tp] = p->bounding_box();
      pair<const float*,int> k(tp.xyz, tp.n);
      TBTable::iterator j = tbtable.find(k);
      if (j == tbtable.end())
	{
	  float *corners = new float[3*8];
	  box_corners(p->bounding_box(), corners);
	  inverse_transform_points(corners, 8, tp.rotation, tp.translation,
				   corners);
	  tbtable[k] = corners;
	}
    }
  else
    p = i->second;

  return p;
}

// ----------------------------------------------------------------------------
//
static void transform_points(const float *xyz, Index n,
			     const float rotation[3][3],
			     const float translation[3],
			     float *txyz)
{
  const float (*r)[3] = rotation;
  for (int64_t k = 0 ; k < n ; ++k)
    {
      Index k3 = 3*k;
      float x = xyz[k3], y = xyz[k3+1], z = xyz[k3+2];
      txyz[k3] = r[0][0]*x + r[0][1]*y + r[0][2]*z + translation[0];
      txyz[k3+1] = r[1][0]*x + r[1][1]*y + r[1][2]*z + translation[1];
      txyz[k3+2] = r[2][0]*x + r[2][1]*y + r[2][2]*z + translation[2];
    }
}

// ----------------------------------------------------------------------------
//
static void inverse_transform_points(const float *xyz, Index n,
				     const float rotation[3][3],
				     const float translation[3],
				     float *itxyz)
{
  const float (*r)[3] = rotation;
  for (int64_t k = 0 ; k < n ; ++k)
    {
      Index k3 = 3*k;
      float x = xyz[k3] - translation[0];
      float y = xyz[k3+1] - translation[1];
      float z = xyz[k3+2] - translation[2];
      itxyz[k3] = r[0][0]*x + r[1][0]*y + r[2][0]*z;
      itxyz[k3+1] = r[0][1]*x + r[1][1]*y + r[2][1]*z;
      itxyz[k3+2] = r[0][2]*x + r[1][2]*y + r[2][2]*z;
    }
}

// ----------------------------------------------------------------------------
//
static void box_corners(const Box &box, float *corners)
{
  float x0 = box.xyz_min[0], y0 = box.xyz_min[1], z0 = box.xyz_min[2];
  float x1 = box.xyz_max[0], y1 = box.xyz_max[1], z1 = box.xyz_max[2];
  corners[0] = x0; corners[1] = y0; corners[2] = z0;
  corners[3] = x0; corners[4] = y0; corners[5] = z1;
  corners[6] = x0; corners[7] = y1; corners[8] = z0;
  corners[9] = x0; corners[10] = y1; corners[11] = z1;
  corners[12] = x1; corners[13] = y0; corners[14] = z0;
  corners[15] = x1; corners[16] = y0; corners[17] = z1;
  corners[18] = x1; corners[19] = y1; corners[20] = z0;
  corners[21] = x1; corners[22] = y1; corners[23] = z1;
}

// ----------------------------------------------------------------------------
//
static void transformed_points_bounding_box(const float *xyz, Index n,
					    const float rotation[3][3],
					    const float translation[3],
					    Box *box)
{
  if (n == 0)
    return;

  const float (*r)[3] = rotation;
  // Initialized to zero to suppress compiler warning about uninitialized use.
  float xmin = 0, ymin = 0, zmin = 0, xmax = 0, ymax = 0, zmax = 0;
  for (int64_t k = 0 ; k < n ; ++k)
    {
      Index k3 = 3*k;
      float x = xyz[k3], y = xyz[k3+1], z = xyz[k3+2];
      float tx = r[0][0]*x + r[0][1]*y + r[0][2]*z + translation[0];
      float ty = r[1][0]*x + r[1][1]*y + r[1][2]*z + translation[1];
      float tz = r[2][0]*x + r[2][1]*y + r[2][2]*z + translation[2];
      if (k == 0)
	{ xmin = xmax = tx; ymin = ymax = ty; zmin = zmax = tz; continue; }
      if (tx < xmin) xmin = tx;
      else if (tx > xmax) xmax = tx;
      if (ty < ymin) ymin = ty;
      else if (ty > ymax) ymax = ty;
      if (tz < zmin) zmin = tz;
      else if (tz > zmax) zmax = tz;
    }
  box->xyz_min[0] = xmin; box->xyz_min[1] = ymin; box->xyz_min[2] = zmin;
  box->xyz_max[0] = xmax; box->xyz_max[1] = ymax; box->xyz_max[2] = zmax;
}

// ----------------------------------------------------------------------------
//
static bool boxes_are_close(const Box &box1, const Box &box2, float distance)
{
  const float *xyz1_min = box1.xyz_min, *xyz1_max = box1.xyz_max;
  const float *xyz2_min = box2.xyz_min, *xyz2_max = box2.xyz_max;
  for (int a = 0 ; a < 3 ; ++a)
    if (xyz1_min[a] > xyz2_max[a] + distance ||
	xyz1_max[a] < xyz2_min[a] - distance)
      return false;

  return true;
}

// ----------------------------------------------------------------------------
//
const char *find_close_points_doc =
  "find_close_points(xyz1, xyz2, max_distance) -> i1, i2\n"
  "\n"
  "Supported API\n"
  "Find close points between two sets of points, returning indices\n"
  "for each point that is close to some point in the other set.\n"
  "It does not report all pairs of close points, only which points\n"
  "have some close neighbor in the other set.\n"
  "The calculation is optimized for specified maximum distance small compared\n"
  "to the extent of the point sets, so that only positions close to\n"
  "a point need to be considered.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "xyz1 : n by 3 float array\n"
  "xyz2 : m by 3 float array\n"
  "max_distance : float\n"
  "  consider only points less than this distance away.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "i1, i2 : numpy int32 arrays\n"
  "  Two arrays of indices into the xyz1 and xyz2 arrays.\n"
  "  The index arrays are generally not of the same length.\n";

// ----------------------------------------------------------------------------
//
extern "C" PyObject *find_close_points(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray xyz1, xyz2;
  double d;
  const char *kwlist[] = {"xyz1", "xyz2", "max_distance", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&d"),
				   (char **)kwlist,
				   parse_float_n3_array, &xyz1,
				   parse_float_n3_array, &xyz2,
				   &d))
    return NULL;

  FArray cxyz1 = xyz1.contiguous_array(), cxyz2 = xyz2.contiguous_array();
  float *scales = NULL;
  Index_List i1, i2;
  find_close_points(CP_BOXES,
		    cxyz1.values(), cxyz1.size(0),
		    cxyz2.values(), cxyz2.size(0),
		    static_cast<float>(d), scales,
		    &i1, &i2);

  return python_tuple(c_array_to_python(i1), c_array_to_python(i2));
}

// ----------------------------------------------------------------------------
//
const char *find_closest_points_doc =
  "find_closest_points(xyz1, xyz2, max_distance [, scale2]) -> i1, i2, near1\n"
  "\n"
  "Supported API\n"
  "Find close points between two sets of points, returning indices\n"
  "for each point that is close to some point in the other set, and\n"
  "also return the closest point in set 2 for each point in set 1.\n"
  "The calculation is optimized for specified maximum distance small compared\n"
  "to the extent of the point sets, so that only positions close to\n"
  "a point need to be considered.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "xyz1 : n by 3 float array\n"
  "xyz2 : m by 3 float array\n"
  "max_distance : float\n"
  "  consider only points less than this distance away.\n"
  "scale2 : length m float array\n"
  "  Optional argument giving a distance scale factor for each point in set 2.\n"
  "  The scale factor is only used for determining the closest point, and\n"
  "  causes the closest scaled distance to be used.  This is an obscure\n"
  "  feature.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "i1, i2 : numpy int32 array\n"
  "  Two arrays of indices into the xyz1 and xyz2 arrays respectively\n"
  "  for points that are within the maximum distance of some other point\n"
  "  in the other array.\n"
  "near1 : numpy int32 array\n"
  "  A third array is returned that gives the index in xyz2\n"
  "  of the closest point for each xyz1 point in the i1 array.\n"
  "  This array has length equal to the length of array i1.\n";

// ----------------------------------------------------------------------------
//
extern "C" PyObject *find_closest_points(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray xyz1, xyz2, scale2;
  double d;
  const char *kwlist[] = {"xyz1", "xyz2", "max_distance", "scale2", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&d|O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &xyz1,
				   parse_float_n3_array, &xyz2,
				   &d,
				   parse_float_n_array, &scale2))
    return NULL;

  if (scale2.dimension() == 1 && scale2.size(0) != xyz2.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "Scales array size does not match points array size");
      return NULL;
    }
  FArray cxyz1 = xyz1.contiguous_array(), cxyz2 = xyz2.contiguous_array();
  FArray sc = scale2.contiguous_array();
  float *sa = (sc.dimension() == 1 ? scale2.values() : NULL);
  Index_List i1, i2, nearest1;
  Py_BEGIN_ALLOW_THREADS
  find_close_points(CP_BOXES,
		    cxyz1.values(), cxyz1.size(0),
		    cxyz2.values(), cxyz2.size(0),
		    static_cast<float>(d), sa,
		    &i1, &i2, &nearest1);
  Py_END_ALLOW_THREADS

  return python_tuple(c_array_to_python(i1), c_array_to_python(i2),
  		      c_array_to_python(nearest1));
}

// ----------------------------------------------------------------------------
//
static bool transformed_points(PyObject *py_tp, Transformed_Points *tp)
{
  if (!PySequence_Check(py_tp))
    {
      PyErr_SetString(PyExc_TypeError,
		      "Transformed points object is not a sequence");
      return false;
    }
  if (PySequence_Length(py_tp) != 2)
    {
      PyErr_SetString(PyExc_TypeError,
		      "Transformed points object is not a sequence of length 2");
      return false;
    }

  PyObject *py_xyz = PySequence_GetItem(py_tp, 0);
  Py_XDECREF(py_xyz);
  float *xyz;
  Index n;
  if (! float_2d_array_values(py_xyz, 3, &xyz, &n))
    return false;

  PyObject *py_xform = PySequence_GetItem(py_tp, 1);
  Py_XDECREF(py_xform);

  float xf[3][4];
  try 
    {
      python_array_to_c(py_xform, &xf[0][0], 3, 4);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return false;
    }

  tp->xyz = xyz;
  tp->n = n/3;
  for (int i = 0 ; i < 3 ; ++i)
    for (int j = 0 ; j < 3 ; ++j)
      tp->rotation[i][j] = xf[i][j];
  for (int i = 0 ; i < 3 ; ++i)
    tp->translation[i] = xf[i][3];

  return true;
}

// ----------------------------------------------------------------------------
//
static bool transformed_points_list(PyObject *py_tpl,
				    vector<Transformed_Points> *tpl)
{
  if (!PySequence_Check(py_tpl))
    {
      PyErr_SetString(PyExc_TypeError,
		      "Transformed points list argument is not a sequence");
      return false;
    }
  Index size = PySequence_Length(py_tpl);
  for (int64_t k = 0 ; k < size ; ++k)
    {
      PyObject *py_tp = PySequence_GetItem(py_tpl, k);
      Py_XDECREF(py_tp);
      Transformed_Points tp;
      if (transformed_points(py_tp, &tp))
	tpl->push_back(tp);
      else
	return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
static PyObject *index_lists(const vector<Index_List> &i)
{
  Index sz = i.size();
  PyObject *t = PyTuple_New(sz);
  for (int64_t k = 0 ; k < sz ; ++k)
    PyTuple_SetItem(t, k, c_array_to_python(i[k]));
  return t;
}

// ----------------------------------------------------------------------------
//
const char *find_close_points_sets_doc =
  "find_close_points_sets(tp1, tp2, max_distance) -> i1, i2\n"
  "\n"
  "Supported API\n"
  "Find close points between two sets of points, returning indices\n"
  "for each point that is close to some point in the other set.\n"
  "Each set of points is defined by a list of pairs, each pair being an\n"
  "array of points and an affine transform to apply to those points.\n"
  "The calculation is optimized for specified maximum distance small compared\n"
  "to the extent of the point sets, so that only positions close to\n"
  "a point need to be considered.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "tp1, tp2 : list of (n by 3 float array, 3 by 4 float array)\n"
  "  The first array of each pair are x,y,z point coordinates,\n"
  "  and the second array is an affine transformation with the\n"
  "  first 3 columns being a 3 by 3 matrix to applied (left multiplied)\n"
  "  to each point and the last column being a translation (added to the point)\n"
  "  applied after the matrix multiplication.\n"
  "max_distance : float\n"
  "  consider only points less than this distance away.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "i1, i2 : tuples of numpy int32 arrays\n"
  "  Two tuples of arrays of indices with lengths matching\n"
  "  the lengths of the tp1 and tp2 arguments. Each index array list\n"
  "  indices into the corresponding transformed point array for points\n"
  "  that have some nearby point in the other set.\n";

// ----------------------------------------------------------------------------
//
extern "C" PyObject *find_close_points_sets(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_tp1, *py_tp2;
  double d;
  const char *kwlist[] = {"tp1", "tp2", "max_distance", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OOd"),
				   (char **)kwlist,
				   &py_tp1, &py_tp2, &d))
    return NULL;

  vector<Transformed_Points> p1, p2;
  if (!transformed_points_list(py_tp1, &p1) ||
      !transformed_points_list(py_tp2, &p2))
    return NULL;

  Index n1 = p1.size(), n2 = p2.size();
  vector<Index_List> i1(n1), i2(n2);
  Py_BEGIN_ALLOW_THREADS
  find_close_points(CP_BOXES, p1, p2, static_cast<float>(d), &i1, &i2);
  Py_END_ALLOW_THREADS

  return python_tuple(index_lists(i1), index_lists(i2));
}
