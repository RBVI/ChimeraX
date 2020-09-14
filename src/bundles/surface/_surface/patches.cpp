/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <map>		// use std::map
#include <iostream>
#include <math.h>			// use sqrt()

#include <arrays/pythonarray.h>		// use python_float_array
#include <arrays/rcarray.h>		// use FArray, IArray

typedef std::vector<float> Vertices;
typedef std::vector<float> Normals;
typedef std::vector<int> Triangles;
typedef std::vector<int> Atoms;		// Atom index for each vertex.
typedef std::vector<int> VertexMap;	// Unique vertex index for each vertex
typedef std::pair<int,int> Edge;
typedef std::map<Edge,int> Edge_Map;

#define min(a,b) (a<b ? a : b)
#define max(a,b) (a>b ? a : b)

inline void add_vertex(Vertices &v, float x, float y, float z)
{
  v.push_back(x);
  v.push_back(y);
  v.push_back(z);
}

inline void add_normal(Normals &n, float nx, float ny, float nz)
{
  n.push_back(nx);
  n.push_back(ny);
  n.push_back(nz);
}

inline void add_triangle(Triangles &t, int v0, int v1, int v2)
{
  if (v0 == v1 || v1 == v2 || v2 == v0)
    {
    std::cerr << "degenerate triangle " << t.size()/3 << " " << v0 << " " << v1 << " " << v2 << std::endl;
    abort();
    }
  t.push_back(v0);
  t.push_back(v1);
  t.push_back(v2);
}

inline int edge_vertex(const Edge_Map &edge_splits, int v1, int v2)
{
  Edge e(v1,v2);
  Edge_Map::const_iterator ei = edge_splits.find(e);
  if (ei == edge_splits.end())
    {
      std::cerr << "attempted to find edge split vertex when non computed " << v1 << " " << v2 << std::endl;
      abort();
    }
  return ei->second;
}

// Find the point on segment p0,p1 equidistant to a0 and a1.
// Returns fraction from p0 to p1.
inline double split_fraction(double x0, double y0, double z0, double x1, double y1, double z1,
			     double ax0, double ay0, double az0, double ax1, double ay1, double az1)
{
  double dx = x1-x0, dy = y1-y0, dz = z1-z0;
  double dax = ax1-ax0, day = ay1-ay0, daz = az1-az0;
  double cx = 0.5*(ax0+ax1)-x0, cy = 0.5*(ay0+ay1)-y0, cz = 0.5*(az0+az1)-z0;
  double dvda = dx*dax + dy*day + dz*daz;
  double dcda = cx*dax + cy*day + cz*daz;
  double f1 = (dvda != 0 ? dcda / dvda : 0.5);
  return f1;
}

// Find the point on segment p0,p1 where the distance to a0 divided by r0 equals
// the distance to a1 divided by r1.  Returns fraction from p0 to p1.
// Requires solving quadratic equation.
inline double scaled_split_fraction(double x0, double y0, double z0, double x1, double y1, double z1,
				    double ax0, double ay0, double az0, double ax1, double ay1, double az1,
				    double r0, double r1)
{
  double dx = x1-x0, dy = y1-y0, dz = z1-z0;
  double d2 = dx*dx + dy*dy + dz*dz;
  double r12 = r1*r1, r02 = r0*r0;
  double a = d2*(r12-r02);
  if (a == 0)
    return split_fraction(x0, y0, z0, x1, y1, z1, ax0, ay0, az0, ax1, ay1, az1);
  double e0x = x0-ax0, e0y = y0-ay0, e0z = z0-az0;
  double e02 = e0x*e0x + e0y*e0y + e0z*e0z;
  double de0 = dx*e0x + dy*e0y + dz*e0z;
  double e1x = x0-ax1, e1y = y0-ay1, e1z = z0-az1;
  double e12 = e1x*e1x + e1y*e1y + e1z*e1z;
  double de1 = dx*e1x + dy*e1y + dz*e1z;

  double b = r12*de0 - r02*de1;
  double c = r12*e02-r02*e12;
  double b2ac = b*b-a*c;
  if (b2ac < 0)
    {
      //      std::cerr << "scaled_split_fraction(): negative discriminant.\n";
      return 0.5;
    }
  double f1 = (-b + sqrt(b2ac))/a;
  return f1;
}

inline void split_point(int v0, int v1, int a0, int a1,
			float *aa, int as0, int as1, float *ra, int rs0, const Vertices &v,
			bool clamp, float *x, float *y, float *z, float *f1)
{
  // Find position to split along edge.
  if (v0 > v1)
    {
      split_point(v1, v0, a1, a0, aa, as0, as1, ra, rs0, v, clamp, x, y, z, f1);
      *f1 = 1 - *f1;
      return;
    }
  // Equidistant from atoms: f = (0.5*(a1xyz+a2xyz)-v0xyz, a1xyz-a0xyz) / (v1xyz-v0xyz, a1xyz-a0xyz)
  float x0 = v[3*v0], y0 = v[3*v0+1], z0 = v[3*v0+2];
  float x1 = v[3*v1], y1 = v[3*v1+1], z1 = v[3*v1+2];
  float ax0 = aa[as0*a0], ay0 = aa[as0*a0+as1], az0 = aa[as0*a0+2*as1];
  float ax1 = aa[as0*a1], ay1 = aa[as0*a1+as1], az1 = aa[as0*a1+2*as1];
  double f = (ra == NULL ?
	     split_fraction(x0,y0,z0, x1,y1,z1, ax0,ay0,az0, ax1,ay1,az1) :
	     scaled_split_fraction(x0,y0,z0, x1,y1,z1, ax0,ay0,az0, ax1,ay1,az1, ra[rs0*a0], ra[rs0*a1]));

  /*
  if (f <= 0 || f >= 1)
    std::cerr << "split_edge(): out of 0-1 range " << f << " edge " << v0 << " " << v1 << " atoms " << a0 << " " << a1 << std::endl;
  */
  if (clamp)
    { if (f < 0) f = 0; else if (f > 1) f = 1; }	// Clamp to 0-1 range.
  double f0 = 1-f;
  *x = f0*x0 + f*x1;
  *y = f0*y0 + f*y1;
  *z = f0*z0 + f*z1;
  *f1 = f;
}

inline int split_edge(int v0, int v1, int a0, int a1,
		      float *aa, int as0, int as1, float *ra, int rs0,
		      Vertices &v, Normals &n, Atoms &va, VertexMap &vm)
{
  float x, y, z, f1;
  split_point(v0, v1, a0, a1, aa, as0, as1, ra, rs0, v, true, &x, &y, &z, &f1);

  // Make vertex at split position
  vm.push_back(v.size()/3);
  add_vertex(v, x, y, z);

  // Make normal at split position
  float f0 = 1-f1;
  float nx = f0*n[3*v0] + f1*n[3*v1];
  float ny = f0*n[3*v0+1] + f1*n[3*v1+1];
  float nz = f0*n[3*v0+2] + f1*n[3*v1+2];
  float n2 = sqrt(nx*nx + ny*ny + nz*nz);
  if (n2 > 0)
	{ nx /= n2; ny /= n2 ; nz /= n2; }
  add_normal(n, nx, ny, nz);

  // Assign atom index for new vertex.
  va.push_back(a0);

  return va.size() - 1;
}

inline void add_split_point(int v0, int v1, int a0, int a1,
			    float *aa, int as0, int as1, float *ra, int rs0,
			    Vertices &v, Normals &n, Atoms &va, VertexMap &vm,
			    Edge_Map &edge_splits)
{
  int vmin = min(v0,v1), vmax = max(v0,v1);
  int amin = (v0 < v1 ? a0 : a1), amax = (v0 < v1 ? a1 : a0);
  Edge e(vmin,vmax);
  if (edge_splits.find(e) == edge_splits.end())
    edge_splits[e] = split_edge(vmin, vmax, amin, amax, aa, as0, as1, ra, rs0, v, n, va, vm);
}

static void compute_edge_split_points(Vertices &v, Normals &n, Atoms &va, VertexMap &vm, const Triangles &t,
				      const FArray &a, const FArray &r, Edge_Map &edge_splits)
{
  int nt = t.size()/3;
  // Get pointers and strides for geometry
  float *aa = a.values();
  int64_t as0 = a.stride(0), as1 = a.stride(1);
  float *ra = (r.dimension() == 1 ? r.values() : NULL);
  int64_t rs0 = (r.dimension() == 1 ? r.stride(0) : 0);
  for (int ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = t[3*ti], v1 = t[3*ti+1], v2 = t[3*ti+2];
      int a0 = va[v0], a1 = va[v1], a2 = va[v2];
      if (a0 != a1)
	add_split_point(v0, v1, a0, a1, aa, as0, as1, ra, rs0, v, n, va, vm, edge_splits);
      if (a1 != a2)
	add_split_point(v1, v2, a1, a2, aa, as0, as1, ra, rs0, v, n, va, vm, edge_splits);
      if (a2 != a0)
	add_split_point(v2, v0, a2, a0, aa, as0, as1, ra, rs0, v, n, va, vm, edge_splits);
    }
}

// Split a triangle along a single cut line, dividing it into 3 new triangles.
inline void cut_triangle_1_line(int v0, int v1, int v2, int a0, int a1, int a2,
				Triangles &t, const Edge_Map &edge_splits)
{
  // Cut line crosses 2 edges.
  // First put in standard orientation with edges 02 and 12 cut.
  if (a1 == a2)
    { int temp = v0; v0 = v1; v1 = v2; v2 = temp; }
  else if (a2 == a0)
    { int temp = v2; v2 = v1; v1 = v0; v0 = temp; }

  // Add 3 triangles to subdivide this one
  int v12, v21, v20, v02;
  v12 = edge_vertex(edge_splits,v1,v2);
  v21 = edge_vertex(edge_splits,v2,v1);
  v20 = edge_vertex(edge_splits,v2,v0);
  v02 = edge_vertex(edge_splits,v0,v2);

  add_triangle(t, v0,v1,v12);
  add_triangle(t, v0,v12,v02);
  add_triangle(t, v2,v20,v21);
}

inline void compute_triple_point(int v0, int v1, int v2, int a0, int a1, int a2, int v01, int v02,
				 Vertices &v, Normals &n, float *aa, int64_t as0, int64_t as1,
				 float *f1, float *f2, float *x, float *y, float *z,
				 float *nx, float *ny, float *nz)
{
  float x0 = v[3*v0], y0 = v[3*v0+1], z0 = v[3*v0+2];
  float x1 = v[3*v1], y1 = v[3*v1+1], z1 = v[3*v1+2];
  float x2 = v[3*v2], y2 = v[3*v2+1], z2 = v[3*v2+2];
  float x01 = x1-x0, y01 = y1-y0, z01 = z1-z0;
  float x02 = x2-x0, y02 = y2-y0, z02 = z2-z0;

  float ax0 = aa[as0*a0], ay0 = aa[as0*a0+as1], az0 = aa[as0*a0+2*as1];
  float ax1 = aa[as0*a1], ay1 = aa[as0*a1+as1], az1 = aa[as0*a1+2*as1];
  float ax2 = aa[as0*a2], ay2 = aa[as0*a2+as1], az2 = aa[as0*a2+2*as1];
  float cx01 = ax1-ax0, cy01 = ay1-ay0, cz01 = az1-az0;
  float cx02 = ax2-ax0, cy02 = ay2-ay0, cz02 = az2-az0;

  float sx01 = v[3*v01], sy01 = v[3*v01+1], sz01 = v[3*v01+2];
  float mx01 = sx01-x0, my01 = sy01-y0, mz01 = sz01-z0;
  float sx02 = v[3*v02], sy02 = v[3*v02+1], sz02 = v[3*v02+2];
  float mx02 = sx02-x0, my02 = sy02-y0, mz02 = sz02-z0;

  float v01c01 = x01*cx01 + y01*cy01 + z01*cz01;
  float v01c02 = x01*cx02 + y01*cy02 + z01*cz02;
  float v02c01 = x02*cx01 + y02*cy01 + z02*cz01;
  float v02c02 = x02*cx02 + y02*cy02 + z02*cz02;
  float m01c01 = mx01*cx01 + my01*cy01 + mz01*cz01;
  float m02c02 = mx02*cx02 + my02*cy02 + mz02*cz02;

  float d = v01c01*v02c02 - v02c01*v01c02;
  float g1 = (d != 0 ? (v02c02*m01c01 - v02c01*m02c02) / d : 0);
  float g2 = (d != 0 ? (v01c01*m02c02 - v01c02*m01c01) / d : 0);
  float nx0 = n[3*v0], ny0 = n[3*v0+1], nz0 = n[3*v0+2];
  float nx1 = n[3*v1], ny1 = n[3*v1+1], nz1 = n[3*v1+2];
  float nx2 = n[3*v2], ny2 = n[3*v2+1], nz2 = n[3*v2+2];
  float nx01 = nx1-nx0, ny01 = ny1-ny0, nz01 = nz1-nz0;
  float nx02 = nx2-nx0, ny02 = ny2-ny0, nz02 = nz2-nz0;
  float nxc = nx0 + g1*nx01 + g2*nx02, nyc = ny0 + g1*ny01 + g2*ny02, nzc = nz0 + g1*nz01 + g2*nz02;
  float n2 = sqrt(nxc*nxc + nyc*nyc + nzc*nzc);
  if (n2 > 0)
    { nxc /= n2; nyc /= n2 ; nzc /= n2; }

  *f1 = g1;
  *f2 = g2;
  *x = x0 + g1*x01 + g2*x02;
  *y = y0 + g1*y01 + g2*y02;
  *z = z0 + g1*z01 + g2*z02;
  *nx = nxc;
  *ny = nyc;
  *nz = nzc;
}

inline void cut_to_vertex(int v0, int v1, int v2, int a1, int a2,
			  int v10, int v12, int v21, int v20,
			  Vertices &v, Normals &n, Atoms &va, VertexMap &vm, Triangles &t)
{
  // Copy edge point.
  int v012 = v.size()/3, v021 = v012 + 1;
  float x0 = v[3*v0], y0 = v[3*v0+1], z0 = v[3*v0+2];
  add_vertex(v, x0, y0, z0);
  add_vertex(v, x0, y0, z0);
  vm.push_back(v0);
  vm.push_back(v0);
  float nx0 = n[3*v0], ny0 = n[3*v0+1], nz0 = n[3*v0+2];
  add_normal(n, nx0, ny0, nz0);
  add_normal(n, nx0, ny0, nz0);
  va.push_back(a1);
  va.push_back(a2);

  add_triangle(t, v012, v10, v12);
  add_triangle(t, v10, v1, v12);
  add_triangle(t, v021, v21, v20);
  add_triangle(t, v2, v20, v21);
}

inline void cut_to_edge(int v0, int v1, int v2, int a2,
			int v01, int v10, int v12, int v21, int v20, int v02,
			Vertices &v, Normals &n, Atoms &va, VertexMap &vm, Triangles &t)
{
  // Copy edge point.
  int v012 = v.size()/3;
  add_vertex(v, v[3*v01], v[3*v01+1], v[3*v01+2]);
  vm.push_back(v01);
  add_normal(n, n[3*v01],  n[3*v01+1], n[3*v01+2]);
  va.push_back(a2);

  add_triangle(t, v0, v01, v02);
  add_triangle(t, v1, v12, v10);
  add_triangle(t, v2, v20, v012);
  add_triangle(t, v2, v012, v21);
}

inline void cut_to_middle(float x, float y, float z, float nx, float ny, float nz,
			  int v0, int v1, int v2, int a0, int a1, int a2,
			  int v01, int v10, int v12, int v21, int v20, int v02,
			  Vertices &v, Normals &n, Atoms &va, VertexMap &vm, Triangles &t)
{
  // Add 3 copies of point and normal to middle of triangle.
  int vn = v.size()/3;
  int vc0 = vn, vc1 = vn+1, vc2 = vn+2;
  for (int c = 0 ; c < 3 ; ++c, ++vn)
    {
      add_vertex(v, x, y, z);
      vm.push_back(vc0);
      add_normal(n, nx, ny, nz);
    }
  va.push_back(a0);
  va.push_back(a1);
  va.push_back(a2);

  // Add 6 triangles to subdivide this one.
  add_triangle(t, v0,v01,vc0);
  add_triangle(t, v0,vc0,v02);
  add_triangle(t, v1,vc1,v10);
  add_triangle(t, v1,v12,vc1);
  add_triangle(t, v2,vc2,v21);
  add_triangle(t, v2,v20,vc2);
}

// Each triangle vertex is closest to a different atom, so the triangle is to be cut into
// 3 regions using 3 cut lines.  It can happen that the intersection of the 3 lines lies
// outside the triangle in which case the triangle is only cut by two lines, a case handled
// by double_cut_triangle().
inline void cut_triangle_3_lines(int v0, int v1, int v2, int a0, int a1, int a2,
				 float *aa, int as0, int as1,
				 Vertices &v, Normals &n, Atoms &va, VertexMap &vm, Triangles &t,
				 const Edge_Map &edge_splits)
{
  // All 3 edges split
  int v01, v10, v12, v21, v20, v02;
  v01 = edge_vertex(edge_splits,v0,v1);
  v10 = edge_vertex(edge_splits,v1,v0);
  v12 = edge_vertex(edge_splits,v1,v2);
  v21 = edge_vertex(edge_splits,v2,v1);
  v20 = edge_vertex(edge_splits,v2,v0);
  v02 = edge_vertex(edge_splits,v0,v2);

  float f1, f2, x, y, z, nx, ny, nz;
  compute_triple_point(v0, v1, v2, a0, a1, a2, v01, v02, v, n, aa, as0, as1,
		       &f1, &f2, &x, &y, &z, &nx, &ny, &nz);

  // Check if triple point is inside triangle.
  float f12 = f1+f2;
  if (f1 > 0 && f2 > 0 && f12 < 1)
    {
      // Point inside triangle.  Divide into 6 new triangles.
      cut_to_middle(x, y, z, nx, ny, nz, v0, v1, v2, a0, a1, a2,
		    v01, v10, v12, v21, v20, v02, v, n, va, vm, t);
    }
  else
    {
      // Triple point lies outside triangle.
      int nout = (f1 > 0 ? 0 : 1) + (f2 > 0 ? 0 : 1) + (f12 < 1 ? 0 : 1);
      if (nout == 1)
	{
	  // Project point to triangle edge.
	  if (f1 <= 0)
	    cut_to_edge(v2, v0, v1, a1, v20, v02, v01, v10, v12, v21, v, n, va, vm, t);
	  else if (f2 <= 0)
	    cut_to_edge(v0, v1, v2, a2, v01, v10, v12, v21, v20, v02, v, n, va, vm, t);
	  else
	    cut_to_edge(v1, v2, v0, a0, v12, v21, v20, v02, v01, v10, v, n, va, vm, t);
	}
      else if (nout == 2)
	{
	  // Project point to triangle vertex.
	  if (f1 > 0)
	    cut_to_vertex(v1, v2, v0, a2, a0, v21, v20, v02, v01, v, n, va, vm, t);
	  else if (f2 > 0)
	    cut_to_vertex(v2, v0, v1, a0, a1, v02, v01, v10, v12, v, n, va, vm, t);
	  else
	    cut_to_vertex(v0, v1, v2, a1, a2, v10, v12, v21, v20, v, n, va, vm, t);
	}
    }
}

inline bool three_patch_edge(const Vertices &v, int v1, int v2, int a1, int a0,
			     float *aa, int64_t as0, int64_t as1, float *ra, int64_t rs0,
			     const Edge_Map &edge_splits, Atoms &va, Edge_Map &edge_3p)
{
  float a0x = aa[as0*a0], a0y = aa[as0*a0+as1], a0z = aa[as0*a0+2*as1];
  float a1x = aa[as0*a1], a1y = aa[as0*a1+as1], a1z = aa[as0*a1+2*as1];
  Edge e(min(v1,v2),max(v1,v2));
  int v12 = edge_vertex(edge_splits,e.first,e.second);
  float v12x = v[3*v12], v12y = v[3*v12+1], v12z = v[3*v12+2];
  float dx = v12x - a0x, dy = v12y - a0y, dz = v12z - a0z;
  float d0 = dx*dx + dy*dy + dz*dz;
  dx = v12x - a1x; dy = v12y - a1y; dz = v12z - a1z;
  float d1 = dx*dx + dy*dy + dz*dz;
  bool div = (d0 < d1);
  if (ra)
    {
      float r0 = ra[rs0*a0], r1 = ra[rs0*a1];
      div = (d0*r1*r1 < d1*r0*r0);
    }
  if (div)
    {
      edge_3p[e] = v12;
      va[v12] = a0;
    }
  return div;
}

inline void split_triangle_1_edge(int v0, int v1, int v2, int v01, Triangles &t)
{
  add_triangle(t, v0, v01, v2);
  add_triangle(t, v1, v2, v01);
}

inline void split_triangle_2_edges(int v0, int v1, int v2, int v01, int v12, Triangles &t)
{
  add_triangle(t, v0, v01, v2);
  add_triangle(t, v1, v12, v01);
  add_triangle(t, v2, v01, v12);
}

inline void split_triangle_3_edges(int v0, int v1, int v2, int v01, int v12, int v20, Triangles &t)
{
  add_triangle(t, v0, v01, v20);
  add_triangle(t, v1, v12, v01);
  add_triangle(t, v2, v20, v12);
  add_triangle(t, v01, v12, v20);
}

inline void split_triangle(int v0, int v1, int v2, int v01, int v12, int v20, Triangles &t)
{
  if (v01 >= 0 && v12 < 0 && v20 < 0)
    split_triangle_1_edge(v0, v1, v2, v01, t);
  else if (v01 < 0 && v12 >= 0 && v20 < 0)
    split_triangle_1_edge(v1, v2, v0, v12, t);
  else if (v01 < 0 && v12 < 0 && v20 >= 0)
    split_triangle_1_edge(v2, v0, v1, v20, t);
  else if (v01 >= 0 && v12 >= 0 && v20 < 0)
    split_triangle_2_edges(v0, v1, v2, v01, v12, t);
  else if (v01 < 0 && v12 >= 0 && v20 >= 0)
    split_triangle_2_edges(v1, v2, v0, v12, v20, t);
  else if (v01 >= 0 && v12 < 0 && v20 >= 0)
    split_triangle_2_edges(v2, v0, v1, v20, v01, t);
  else if (v01 >= 0 && v12 >= 0 && v20 >= 0)
    split_triangle_3_edges(v0, v1, v2, v01, v12, v20, t);
}

inline int minimize_atom_distance(int v0, int v1, const Vertices &v, int a0, int a1,
				  float *aa, int64_t as0, int64_t as1, float *ra, int64_t rs0,
				  Atoms &va)
{
  if (a0 == a1)
    return 0;

  float a0x = aa[as0*a0], a0y = aa[as0*a0+as1], a0z = aa[as0*a0+2*as1];
  float a1x = aa[as0*a1], a1y = aa[as0*a1+as1], a1z = aa[as0*a1+2*as1];
  float v0x = v[3*v0], v0y = v[3*v0+1], v0z = v[3*v0+2];
  float v1x = v[3*v1], v1y = v[3*v1+1], v1z = v[3*v1+2];

  double dx, dy, dz;
  dx = v1x - a0x; dy = v1y - a0y; dz = v1z - a0z;
  double d10 = dx*dx + dy*dy + dz*dz;
  dx = v1x - a1x; dy = v1y - a1y; dz = v1z - a1z;
  double d11 = dx*dx + dy*dy + dz*dz;
  dx = v0x - a0x; dy = v0y - a0y; dz = v0z - a0z;
  double d00 = dx*dx + dy*dy + dz*dz;
  dx = v0x - a1x; dy = v0y - a1y; dz = v0z - a1z;
  double d01 = dx*dx + dy*dy + dz*dz;
 
  int change = 0;
  if (ra)
    {
      double r0 = ra[rs0*a0], r1 = ra[rs0*a1];
      if (d10*r1*r1 < d11*r0*r0)
	{ va[v1] = a0; change += 1; }
      if (d01*r0*r0 < d00*r1*r1)
	{ va[v0] = a1; change += 1; }
    }
   else
    {
      if (d10 < d11)
	{ va[v1] = a0; change += 1; }
      if (d01 < d00)
	{ va[v0] = a1; change += 1; }
    }

  float x, y, z, f;
  split_point(v0, v1, va[v0], va[v1], aa, as0, as1, ra, rs0, v, false, &x, &y, &z, &f);
  //  if (f <= -0.01 || f >= 1.01)
  /*
  if (f <= 0 || f >= 1)
    {
    std::cerr << "minimize_atom_distance(): inconsitent with split " << f << " edge " << v0 << " " << v1 << " atoms " << a0 << " " << a1 << std::endl;
    std::cerr << "edge vector " << v[3*v0]-v[3*v1] << " " << v[3*v0+1]-v[3*v1+1] << " " << v[3*v0+2]-v[3*v1+2] << std::endl;
    }
  */
  return change;
}

static int minimize_atom_distances(const Vertices &v, Atoms &va, const Triangles &t,
				   const FArray &a, const FArray &r)
{
  int nt = t.size()/3;
  float *aa = a.values();
  int64_t as0 = a.stride(0), as1 = a.stride(1);
  float *ra = (r.dimension() == 1 ? r.values() : NULL);
  int64_t rs0 = (r.dimension() == 1 ? r.stride(0) : 0);
  int change = 0;
  for (int64_t ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = t[3*ti], v1 = t[3*ti+1], v2 = t[3*ti+2];
      int a0 = va[v0], a1 = va[v1], a2 = va[v2];
      change += minimize_atom_distance(v0, v1, v, a0, a1, aa, as0, as1, ra, rs0, va);
      change += minimize_atom_distance(v1, v2, v, a1, a2, aa, as0, as1, ra, rs0, va);
      change += minimize_atom_distance(v2, v0, v, a2, a0, aa, as0, as1, ra, rs0, va);
    }
  return change;
}

static void find_3_patch_edges(const Vertices &v, const Triangles &t, Atoms &va,
			       const FArray &a, const FArray &r,
			       const Edge_Map &edge_splits, Edge_Map &edge_3p)
{
  int nt = t.size()/3;
  float *aa = a.values();
  int64_t as0 = a.stride(0), as1 = a.stride(1);
  float *ra = (r.dimension() == 1 ? r.values() : NULL);
  int64_t rs0 = (r.dimension() == 1 ? r.stride(0) : 0);
  for (int64_t ti = 0 ; ti < nt ; ++ti)
     {
      int v0 = t[3*ti], v1 = t[3*ti+1], v2 = t[3*ti+2];
      int a0 = va[v0], a1 = va[v1], a2 = va[v2];
      if (a0 != a1 && a1 != a2 && a2 != a0)
	{
	  three_patch_edge(v, v0, v1, a0, a2, aa, as0, as1, ra, rs0, edge_splits, va, edge_3p);
	  three_patch_edge(v, v1, v2, a1, a0, aa, as0, as1, ra, rs0, edge_splits, va, edge_3p);
	  three_patch_edge(v, v2, v0, a2, a1, aa, as0, as1, ra, rs0, edge_splits, va, edge_3p);
	}
     }
}

static void split_3_patch_triangles(const Triangles &t, const Atoms &va, const Edge_Map &edge_3p,
				    Triangles &tsplit, Triangles &tunsplit)
{
  Edge_Map::const_iterator ei;
  int64_t nt = t.size()/3;
  for (int64_t ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = t[3*ti], v1 = t[3*ti+1], v2 = t[3*ti+2];
      int a0 = va[v0], a1 = va[v1], a2 = va[v2];
      int v01 = -1, v12 = -1, v20 = -1;
      if (a0 != a1 && (ei = edge_3p.find(Edge(min(v0,v1),max(v0,v1))), ei != edge_3p.end()))
	v01 = ei->second;
      if (a1 != a2 && (ei = edge_3p.find(Edge(min(v1,v2),max(v1,v2))), ei != edge_3p.end()))
	v12 = ei->second;
      if (a2 != a0 && (ei = edge_3p.find(Edge(min(v2,v0),max(v2,v0))), ei != edge_3p.end()))
	v20 = ei->second;
      if (v01 >= 0 || v12 >= 0 || v20 >= 0)
	split_triangle(v0, v1, v2, v01, v12, v20, tsplit);
      else
	add_triangle(tunsplit, v0, v1, v2);
    }
}

static int refine_3_patch_triangles(Vertices &v, Normals &n, Triangles &t, Atoms &va, VertexMap &vm,
				    const FArray &a, const FArray &r, Edge_Map &edge_splits)
{
  Edge_Map edge_3p;
  find_3_patch_edges(v, t, va, a, r, edge_splits, edge_3p);
  //  std::cerr << "Three patch edges: " << edge_3p.size() << std::endl;

  Triangles tsplit, tunsplit;
  split_3_patch_triangles(t, va, edge_3p, tsplit, tunsplit);
  //  std::cerr << "Triangle splits produced " << tsplit.size()/3 << " triangles\n";

  // Remove edge split points that are now vertices of triangles.
  for (Edge_Map::iterator ei = edge_3p.begin() ; ei != edge_3p.end() ; ++ei)
    edge_splits.erase(ei->first);
  edge_3p.clear();

  // Fix closest atom assignments.
  for (int i = 0 ; i < 10 ; ++i)
    {
      int change = minimize_atom_distances(v, va, tsplit, a, r);
      change += minimize_atom_distances(v, va, tunsplit, a, r);
      if (change == 0)
	break;
      //      std::cerr << "Fixed " << change << " atom assignments\n";
    }

  compute_edge_split_points(v, n, va, vm, tsplit, a, r, edge_splits);
  // The unsplit can also have edges that need split because vertex was assigned a new atom.
  compute_edge_split_points(v, n, va, vm, tunsplit, a, r, edge_splits);

  t.clear();
  t.insert(t.end(), tunsplit.begin(), tunsplit.end());
  t.insert(t.end(), tsplit.begin(), tsplit.end());

  return tsplit.size()/3;
}

static void duplicate_edge_vertices(Vertices &v, Normals &n, Atoms &va, VertexMap &vm, Edge_Map &edge_splits)
{
  // Duplicate split points.
  Edge_Map edup;
  int vn = v.size()/3;
  for (Edge_Map::iterator ei = edge_splits.begin() ; ei != edge_splits.end() ; ++ei)
    {
      Edge e = ei->first;
      int ev = ei->second;
      edup[Edge(e.second,e.first)] = vn;
      add_vertex(v, v[3*ev], v[3*ev+1], v[3*ev+2]);
      vm.push_back(ev);
      add_normal(n, n[3*ev], n[3*ev+1], n[3*ev+2]);
      va.push_back(va[e.second]);
      va[ev] = va[e.first];		// Update atom assignment changed in case vertex 1 assignment changed.
      vn += 1;
    }
  edge_splits.insert(edup.begin(), edup.end());
}

static void divide_triangles(Vertices &v, Normals &n, Triangles &t, Atoms &va, VertexMap &vm,
			     const FArray &a, Edge_Map &edge_splits)
{
  duplicate_edge_vertices(v, n, va, vm, edge_splits);

  int nt = t.size()/3;
  float *aa = a.values();
  int64_t as0 = a.stride(0), as1 = a.stride(1);
  Triangles td;
  for (int64_t ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = t[3*ti], v1 = t[3*ti+1], v2 = t[3*ti+2];
      int a0 = va[v0], a1 = va[v1], a2 = va[v2];
      if (a0 == a1 && a1 == a2)
	add_triangle(td, v0, v1, v2);  // copy triangle, no subdivision
      else if (a0 != a1 && a1 != a2 && a2 != a0)
	cut_triangle_3_lines(v0, v1, v2, a0, a1, a2, aa, as0, as1,
			     v, n, va, vm, td, edge_splits);
      else
	// Cut triangle along one line.
	cut_triangle_1_line(v0, v1, v2, a0, a1, a2, td, edge_splits);
    }
  t.clear();
  t.insert(t.end(), td.begin(), td.end());
}

static void convert_arrays_to_vectors(const FArray &vs, const FArray &ns,
				      const IArray &vas, const IArray &ts,
				      Vertices &v, Normals &n, Atoms &va, VertexMap &vm, Triangles &t)
{
  // Get pointers and strides for arrays.
  float *vsa = vs.values(), *nsa = ns.values();
  int *vasa = vas.values();
  int64_t vs0 = vs.stride(0), vs1 = vs.stride(1);
  int64_t ns0 = ns.stride(0), ns1 = ns.stride(1);
  int64_t vas0 = vas.stride(0);

  // Copy vertices and normals to vectors.
  int nv = vs.size(0);
  for (int i = 0 ; i < nv ; ++i)
    {
      add_vertex(v, vsa[i*vs0], vsa[i*vs0+vs1], vsa[i*vs0+2*vs1]);
      vm.push_back(i);
      add_normal(n, nsa[i*ns0], nsa[i*ns0+ns1], nsa[i*ns0+2*ns1]);
      va.push_back(vasa[i*vas0]);
    }

  // Copy triangles array to vector.
  int nt = ts.size(0);
  int *tsa = ts.values();
  int64_t ts0 = ts.stride(0), ts1 = ts.stride(1);
  for (int i = 0 ; i < nt ; ++i)
    add_triangle(t, tsa[ts0*i], tsa[ts0*i+ts1], tsa[ts0*i+2*ts1]);
}

static void sharp_patches(Vertices &v, Normals &n, Triangles &t, Atoms &va, VertexMap &vm,
			  const FArray &a, const FArray &r, int refinement_steps)
{
  // Make vertices for split edges.
  Edge_Map edge_splits;
  compute_edge_split_points(v, n, va, vm, t, a, r, edge_splits);

  // Split edges that span three atom zones for better boundaries
  // for narrow strip atom zones.
  for (int i = 0 ; i < refinement_steps ; ++i)
    if (refine_3_patch_triangles(v, n, t, va, vm, a, r, edge_splits) == 0)
      break;

  // Make subdivided triangles along atom zone boundaries.
  divide_triangles(v, n, t, va, vm, a, edge_splits);
}
			  
// ----------------------------------------------------------------------------
//
extern "C" PyObject *sharp_edge_patches(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray vertices, normals, axyz, radii;
  IArray triangles, vertex_atoms;
  int refinement_steps = 0;
  const char *kwlist[] = {"vertices", "normals", "triangles", "vertex_atoms", "atom_xyz", "atom_radii",
			  "refinement_steps", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&O&|O&i"), (char **)kwlist,
				   parse_float_n3_array, &vertices,
				   parse_float_n3_array, &normals,
				   parse_int_n3_array, &triangles,
				   parse_int_n_array, &vertex_atoms,
				   parse_float_n3_array, &axyz,
				   parse_float_n_array, &radii,
				   &refinement_steps))
    return NULL;

  if (normals.size(0) != vertices.size(0))
    {
      PyErr_SetString(PyExc_ValueError, "normals and vertices have different lengths");
      return NULL;
    }
  if (vertex_atoms.size(0) != vertices.size(0))
    {
      PyErr_SetString(PyExc_ValueError, "vertex map and vertices have different sizes");
      return NULL;
    }
  if (radii.dimension() == 1 && radii.size(0) != axyz.size(0))
    {
      PyErr_SetString(PyExc_ValueError, "atom coordinates and radii arrays have different sizes");
      return NULL;
    }

  Vertices v;
  Normals n;
  Triangles t;
  Atoms va;
  VertexMap vm;
  Py_BEGIN_ALLOW_THREADS
    convert_arrays_to_vectors(vertices, normals, vertex_atoms, triangles, v, n, va, vm, t);
    sharp_patches(v, n, t, va, vm, axyz, radii, refinement_steps);
  Py_END_ALLOW_THREADS

  int nv = v.size()/3, nt = t.size()/3;
  PyObject *vp = c_array_to_python(v, nv, 3);
  PyObject *np = c_array_to_python(n, nv, 3);
  PyObject *tp = c_array_to_python(t, nt, 3);
  PyObject *vap = c_array_to_python(va);

  // Compute triangles without duplicated vertices for surface calculations requiring no boundary.
  for (int i = 0 ; i < nv ; ++i)
    if (vm[i] != i && vm[vm[i]] != vm[i])
      vm[i] = vm[vm[i]];  // Map vertices duplicated two times.
  int *tja, nt3 = 3*nt;
  PyObject *tj = python_int_array(nt, 3, &tja);
  for (int i = 0 ; i < nt3 ; ++i)
    tja[i] = vm[t[i]];

  PyObject *r = python_tuple(vp, np, tp, tj, vap);

  for (int64_t ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = t[3*ti], v1 = t[3*ti+1], v2 = t[3*ti+2];
      int a0 = va[v0], a1 = va[v1], a2 = va[v2];
      if (a0 != a1 || a1 != a2 || a2 != a0)
	std::cerr << "multi color tri " << v0 << " "  << v1 << " "  << v2 << " "  << a0 << " "  << a1 << " "  << a2 << std::endl;
    }


  return r;
}

class Vertex
{
public:
  Vertex(float x, float y, float z): x(x), y(y), z(z) {}
  bool operator<(const Vertex &v) const
  { return x < v.x || (x == v.x && (y < v.y || (y == v.y && z < v.z))); }
private:
  float x,y,z;
};

// ----------------------------------------------------------------------------
// This code is slow, 8 seconds for 16 million vertices, late 2012 iMac.
// stl::unordered_map is twice as fast in tests, but that is still slow.
//
static void unique_vertices(const FArray &vertices, int *vmap)
{
  std::map<Vertex,int> vm;
  int nv = vertices.size(0);
  int64_t vs0 = vertices.stride(0), vs1 = vertices.stride(1);
  const float *va = vertices.values();
  for (int v = 0 ; v < nv ; ++v)
    {
      Vertex p(va[vs0*v], va[vs0*v+vs1], va[vs0*v+2*vs1]);
      std::map<Vertex,int>::iterator vi = vm.find(p);
      if (vi == vm.end())
	{
	  vm[p] = v;
	  vmap[v] = v;
	}
      else
	vmap[v] = vi->second;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *unique_vertex_map(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray vertices;
  const char *kwlist[] = {"vertices", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"), (char **)kwlist,
				   parse_float_n3_array, &vertices))
    return NULL;

  int *vmap;
  PyObject *vm = python_int_array(vertices.size(0), &vmap);
  Py_BEGIN_ALLOW_THREADS
  unique_vertices(vertices, vmap);
  Py_END_ALLOW_THREADS
  return vm;
}
