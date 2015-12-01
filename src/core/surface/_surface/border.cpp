// ----------------------------------------------------------------------------
// Compute the loops resulting from a plane intersected with a surface.
//
//#include <iostream>			// use std::cerr for debugging
#include <map>				// use std::map
#include <vector>			// use std::vector

#include "border.h"			// Use Vertices, Loops
#include "rcarray.h"			// use FArray, IArray

namespace Cap_Calculation
{

// ----------------------------------------------------------------------------
// Pair of indices where first is always <= second.
//
class Index_Pair : public std::pair<int, int>
{
 public:
  Index_Pair() { this->first = 0; this->second = 0; }
  Index_Pair(int i, int j)
    {
      if (i <= j)
	{ this->first = i; this->second = j; }
      else
	{ this->first = j; this->second = i; }
    }
};

typedef std::map<Index_Pair, Index_Pair> Edge_Map;

// ----------------------------------------------------------------------------
//
static void side_of_plane(float plane_normal[3], float plane_offset,
			  const FArray &varray, std::vector<float> &side);
static void calculate_plane_edges(const IArray &tarray,
				  const std::vector<float> &side,
				  Edge_Map &edges);
static void calculate_loops(Edge_Map &edges, const FArray &varray,
			    const std::vector<float> &side,
			    Vertices &points, Loops &loops);
static void add_plane_point(const Index_Pair &e0,
			    const std::vector<float> &side,
			    float *v, Vertices &points);

// ----------------------------------------------------------------------------
// Loops are formed by a consecutive sequence of vertex indices.
// The loop index pair gives the start and end of the vertices for the loop.
//
void calculate_border(float plane_normal[3], float plane_offset,
		      const FArray &varray, const IArray &tarray, /* Surface */
		      Vertices &border_vertices, Loops &loops)
{
  // Find which side of plane each surface vertex lies on.
  std::vector<float> side;
  side_of_plane(plane_normal, plane_offset, varray, side);

  // Find surface triangles that intersect plane and record planar edges.
  Edge_Map edges;
  calculate_plane_edges(tarray, side, edges);

  // Calculate loops in plane and xyz vertex positions in plane.
  calculate_loops(edges, varray, side, border_vertices, loops);
}

// ----------------------------------------------------------------------------
// Calculate which side of plane each point is on.
//
static void side_of_plane(float plane_normal[3], float plane_offset,
			  const FArray &varray, std::vector<float> &side)
{
  int n = varray.size(0);
  side.assign(n, 0.0);
  FArray cvarray = varray.contiguous_array();
  float *v = cvarray.values();
  float nx = plane_normal[0], ny = plane_normal[1], nz = plane_normal[2];
  for (int k = 0 ; k < n ; ++k)
    {
      int k3 = 3 * k;
      side[k] = nx * v[k3] + ny * v[k3+1] + nz * v[k3+2] - plane_offset;
    }
}

// ----------------------------------------------------------------------------
// Calculate which side of plane each point is on.
//
static void calculate_plane_edges(const IArray &tarray,
				  const std::vector<float> &side,
				  Edge_Map &edges)
{
  int m = tarray.size(0);
  IArray ctarray = tarray.contiguous_array();
  int *t = ctarray.values();
  for (int k = 0 ; k < m ; ++k)
    {
      int k3 = 3 * k;
      int i0 = t[k3], i1 = t[k3+1], i2 = t[k3+2];
      if (i0 == i1 || i0 == i2 || i1 == i2)
	// Ignore degenerate triangles to avoid more than 2 triangles sharing an edge.
	continue;
      float s0 = side[i0], s1 = side[i1], s2 = side[i2];
      int c0, c1, c2;
      if (s0 >= 0 && s1 < 0 && s2 < 0)
	{ c0 = i1; c1 = i2; c2 = i0; }
      else if (s0 < 0 && s1 >= 0 && s2 >= 0)
	{ c0 = i2; c1 = i1; c2 = i0; }
      else if (s1 >= 0 && s0 < 0 && s2 < 0)
	{ c0 = i2; c1 = i0; c2 = i1; }
      else if (s1 < 0 && s0 >= 0 && s2 >= 0)
	{ c0 = i0; c1 = i2; c2 = i1; }
      else if (s2 >= 0 && s0 < 0 && s1 < 0)
	{ c0 = i0; c1 = i1; c2 = i2; }
      else if (s2 < 0 && s0 >= 0 && s1 >= 0)
	{ c0 = i1; c1 = i0; c2 = i2; }
      else
	continue;	// No intersection
      // TODO: Assuming that two triangles don't traverse edge in same
      // direction.  Need to catch that.
      Index_Pair v0(c0,c2), v1(c1,c2);
      edges[v0] = v1;
    }
}

// ----------------------------------------------------------------------------
// Calculate loops in plane and xyz vertex positions in plane.
//
static void calculate_loops(Edge_Map &edges, const FArray &varray,
			    const std::vector<float> &side,
			    std::vector<float> &points,
			    std::vector<Loop> &loops)
{
  FArray cvarray = varray.contiguous_array();
  float *v = cvarray.values();

  int nc = 0;
  int start = 0, next = 0;
  while (edges.size() > 0)
    {
      // Pop an edge and trace it.
      // Should I try to preserve non-loops?  Throw them away for now.
      Edge_Map::iterator e = edges.begin();
      Index_Pair start_edge = e->first;
      while (e != edges.end())
	{
	  Index_Pair e0 = e->first, e1 = e->second;
	  edges.erase(e);
	  // calculate point for e0 and add to points vector.
	  add_plane_point(e0, side, v, points);
	  next += 1;
	  e = edges.find(e1);
	  if (e == edges.end())
	    {
	    if (e1 == start_edge)
	      {
		//		std::cerr << "Found loop " << start << " - " << next << std::endl;
		// finished loop
		loops.push_back(Loop(start, next-1));
		start = next;
	      }
	    else
	      {
		//		std::cerr << "Erasing non-loop " << start << " - " << next << std::endl;
		// Non-loop.  Erase points back to start of trace.
		points.erase(points.begin()+3*start, points.end());
		next = start;
		nc += 1;
	      }
	    }
	}
    }
  //  if (nc > 0)
  //    std::cerr << " found non-closed loops on plane\n";
}

// ----------------------------------------------------------------------------
// Compute the intersection point of a surface triangle edge with the plane.
//
static void add_plane_point(const Index_Pair &e0,
			    const std::vector<float> &side,
			    float *v, std::vector<float> &points)
{
  int i0 = e0.first, i1 = e0.second;
  float s0 = side[i0], s1 = side[i1];
  float f0 = s1 / (s1 - s0), f1 = -s0 / (s1 - s0);
  float *v0 = &(v[3*i0]), *v1 = &(v[3*i1]);
  float x = f0 * v0[0] + f1 * v1[0];
  float y = f0 * v0[1] + f1 * v1[1];
  float z = f0 * v0[2] + f1 * v1[2];
  points.push_back(x);
  points.push_back(y);
  points.push_back(z);
}

}	// end of namespace Cap_Calculation
