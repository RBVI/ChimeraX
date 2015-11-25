#include <map>		// use std::map
#include <iostream>
#include <math.h>			// use sqrt()

#include "pythonarray.h"		// use python_float_array
#include "rcarray.h"			// use FArray, IArray

typedef std::vector<float> Vertices;
typedef std::vector<float> Normals;
typedef std::vector<int> Triangles;
typedef std::vector<int> Atoms;
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

inline int edge_vertex(Edge_Map &edge_splits, int v1, int v2)
{
  Edge e(v1,v2);
  Edge_Map::iterator ei = edge_splits.find(e);
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
			float *aa, int as0, int as1, float *ra, int rs0, Vertices &vs,
			bool clamp, float *x, float *y, float *z, float *f1)
{
  // Find position to split along edge.
  if (v0 > v1)
    {
      split_point(v1, v0, a1, a0, aa, as0, as1, ra, rs0, vs, clamp, x, y, z, f1);
      *f1 = 1 - *f1;
      return;
    }
  // Equidistant from atoms: f = (0.5*(a1xyz+a2xyz)-v0xyz, a1xyz-a0xyz) / (v1xyz-v0xyz, a1xyz-a0xyz)
  float x0 = vs[3*v0], y0 = vs[3*v0+1], z0 = vs[3*v0+2];
  float x1 = vs[3*v1], y1 = vs[3*v1+1], z1 = vs[3*v1+2];
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

inline int split_edge(int v0, int v1, int a0, int a1, float *aa, int as0, int as1, float *ra, int rs0,
		      Vertices &vs, Normals &ns, Atoms &v2as)
{
  float x, y, z, f1;
  split_point(v0, v1, a0, a1, aa, as0, as1, ra, rs0, vs, true, &x, &y, &z, &f1);

  // Make vertex at split position
  add_vertex(vs, x, y, z);

  // Make normal at split position
  float f0 = 1-f1;
  float nx = f0*ns[3*v0] + f1*ns[3*v1];
  float ny = f0*ns[3*v0+1] + f1*ns[3*v1+1];
  float nz = f0*ns[3*v0+2] + f1*ns[3*v1+2];
  float n2 = sqrt(nx*nx + ny*ny + nz*nz);
  if (n2 > 0)
	{ nx /= n2; ny /= n2 ; nz /= n2; }
  add_normal(ns, nx, ny, nz);

  // Assign atom index for new vertex.
  v2as.push_back(a0);

  return v2as.size() - 1;
}

inline void add_split_point(int v0, int v1, int a0, int a1,
			    float *aa, int as0, int as1, float *ra, int rs0,
			    Vertices &vs, Normals &ns,
			    Atoms &v2as, Edge_Map &edge_splits)
{
  int vmin = min(v0,v1), vmax = max(v0,v1);
  int amin = (v0 < v1 ? a0 : a1), amax = (v0 < v1 ? a1 : a0);
  Edge e(vmin,vmax);
  if (edge_splits.find(e) == edge_splits.end())
    edge_splits[e] = split_edge(vmin, vmax, amin, amax, aa, as0, as1, ra, rs0, vs, ns, v2as);
}

static void compute_edge_split_points(Vertices &vs, Normals &ns, const Triangles &t,
				      Atoms &v2as, const FArray &a, const FArray &r,
				      Edge_Map &edge_splits)
{
  int nt = t.size()/3;
  // Get pointers and strides for geometry
  float *aa = a.values();
  long as0 = a.stride(0), as1 = a.stride(1);
  float *ra = (r.dimension() == 1 ? r.values() : NULL);
  long rs0 = (r.dimension() == 1 ? r.stride(0) : 0);
  for (int ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = t[3*ti], v1 = t[3*ti+1], v2 = t[3*ti+2];
      int a0 = v2as[v0], a1 = v2as[v1], a2 = v2as[v2];
      if (a0 != a1)
	add_split_point(v0, v1, a0, a1, aa, as0, as1, ra, rs0, vs, ns, v2as, edge_splits);
      if (a1 != a2)
	add_split_point(v1, v2, a1, a2, aa, as0, as1, ra, rs0, vs, ns, v2as, edge_splits);
      if (a2 != a0)
	add_split_point(v2, v0, a2, a0, aa, as0, as1, ra, rs0, vs, ns, v2as, edge_splits);
    }
}

// Split a triangle along a single cut line, dividing it into 3 new triangles.
inline void cut_triangle_1_line(int v0, int v1, int v2, int a0, int a1, int a2,
				Triangles &ts, Edge_Map &edge_splits)
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

  add_triangle(ts, v0,v1,v12);
  add_triangle(ts, v0,v12,v02);
  add_triangle(ts, v2,v20,v21);
}

// Each triangle vertex is closest to a different atom.  Normally this would partition
// the triangle into 3 regions using 3 lines that intersect at a point inside the triangle.
// But if the intersection point lies outside the triangle then it is only cut by 2 lines.
// To test for this case see if the cut point along edge 12 is closer to the atom of vertex 0
// than the atoms of vertex 1 and 2.
//
// Returns true if triangle is double cut, otherwise false.
//
inline bool cut_triangle_2_lines(int v0, int v1, int v2, int a0, int a1,
				 int v01, int v10, int v12, int v21, int v20, int v02,
				 float *aa, int as0, int as1, float *ra, int rs0,
				 Vertices &vs, Normals &ns,
				 Triangles &ts, Atoms &v2as)
{
  float ex0 = vs[3*v12], ey0 = vs[3*v12+1], ez0 = vs[3*v12+2];

  float ax0 = aa[as0*a0], ay0 = aa[as0*a0+as1], az0 = aa[as0*a0+2*as1];
  float ax1 = aa[as0*a1], ay1 = aa[as0*a1+as1], az1 = aa[as0*a1+2*as1];
  //  float ax2 = aa[as0*a2], ay2 = aa[as0*a2+as1], az2 = aa[as0*a2+2*as1];

  float dx0 = ex0-ax0, dy0 = ey0-ay0, dz0 = ez0-az0;
  float dx1 = ex0-ax1, dy1 = ey0-ay1, dz1 = ez0-az1;
  float d0 = dx0*dx0 + dy0*dy0 + dz0*dz0;
  float d1 = dx1*dx1 + dy1*dy1 + dz1*dz1;
  if (ra)
    {
      float r0 = ra[a0*rs0], r1 = ra[a1*rs0];
      if (d0*r1*r1 >= d1*r0*r0)
	return false;
    }
  else if (d0 >= d1)
    return false;

  // Double cut edge 12.
  /*
  int v12a = split_edge(v1, v2, a1, a0, va, vs0, vs1, na, ns0, ns1, aa, as0, as1, ra, rs0, vs, ns, v2as);
  int v21a = v12a + 1;
  int v12b = split_edge(v1, v2, a0, a2, va, vs0, vs1, na, ns0, ns1, aa, as0, as1, ra, rs0, vs, ns, v2as);
  int v21b = v12b + 1;
  add_triangle(ts, v1, v12a, v10);
  add_triangle(ts, v2, v20, v21b);
  add_triangle(ts, v0, v01, v21a);
  add_triangle(ts, v0, v21a, v12b);
  add_triangle(ts, v0, v12b, v02);
  */

  int v102 = vs.size()/3;
  add_vertex(vs, ex0, ey0, ez0);
  float nx = ns[3*v12], ny = ns[3*v12+1], nz = ns[3*v12+2];
  add_normal(ns, nx, ny, nz);
  v2as.push_back(a0);

  add_triangle(ts, v0, v01, v102);
  add_triangle(ts, v0, v102, v02);
  add_triangle(ts, v1, v12, v10);
  add_triangle(ts, v2, v20, v21);

  /*
    For iterative subdivision.  This requires used of fix_closest_atoms().
  (*v2as)[v21] = a0;
  (*v2as)[v12] = a0;
  */

  //  std::cerr << "double cut " << v1 << " " << v2 << " " << a1 << " " << a2 << " " << a0 << std::endl;

  return true;
}

inline void compute_triple_point(int v0, int v1, int v2, int a0, int a1, int a2, int v01, int v02,
				 Vertices &vs, Normals &ns, float *aa, long as0, long as1,
				 float *f1, float *f2, float *x, float *y, float *z,
				 float *nx, float *ny, float *nz)
{
  float x0 = vs[3*v0], y0 = vs[3*v0+1], z0 = vs[3*v0+2];
  float x1 = vs[3*v1], y1 = vs[3*v1+1], z1 = vs[3*v1+2];
  float x2 = vs[3*v2], y2 = vs[3*v2+1], z2 = vs[3*v2+2];
  float x01 = x1-x0, y01 = y1-y0, z01 = z1-z0;
  float x02 = x2-x0, y02 = y2-y0, z02 = z2-z0;

  float ax0 = aa[as0*a0], ay0 = aa[as0*a0+as1], az0 = aa[as0*a0+2*as1];
  float ax1 = aa[as0*a1], ay1 = aa[as0*a1+as1], az1 = aa[as0*a1+2*as1];
  float ax2 = aa[as0*a2], ay2 = aa[as0*a2+as1], az2 = aa[as0*a2+2*as1];
  float cx01 = ax1-ax0, cy01 = ay1-ay0, cz01 = az1-az0;
  float cx02 = ax2-ax0, cy02 = ay2-ay0, cz02 = az2-az0;

  //	  float cx12 = ax2-ax1, cy12 = ay2-ay1, cz12 = az2-az1;
  float sx01 = vs[3*v01], sy01 = vs[3*v01+1], sz01 = vs[3*v01+2];
  float mx01 = sx01-x0, my01 = sy01-y0, mz01 = sz01-z0;
  //  float mx01 = 0.5*(ax0+ax1)-x0, my01 = 0.5*(ay0+ay1)-y0, mz01 = 0.5*(az0+az1)-z0;
  float sx02 = vs[3*v02], sy02 = vs[3*v02+1], sz02 = vs[3*v02+2];
  float mx02 = sx02-x0, my02 = sy02-y0, mz02 = sz02-z0;
  //  float mx02 = 0.5*(ax0+ax2)-x0, my02 = 0.5*(ay0+ay2)-y0, mz02 = 0.5*(az0+az2)-z0;
  //	  float mx12 = 0.5*(ax1+ax2)-x0, my12 = 0.5*(ay1+ay2)-y0, mz12 = 0.5*(az1+az2)-z0;

  float v01c01 = x01*cx01 + y01*cy01 + z01*cz01;
  float v01c02 = x01*cx02 + y01*cy02 + z01*cz02;
  float v02c01 = x02*cx01 + y02*cy01 + z02*cz01;
  float v02c02 = x02*cx02 + y02*cy02 + z02*cz02;
  float m01c01 = mx01*cx01 + my01*cy01 + mz01*cz01;
  float m02c02 = mx02*cx02 + my02*cy02 + mz02*cz02;

  float d = v01c01*v02c02 - v02c01*v01c02;
  float g1 = (d != 0 ? (v02c02*m01c01 - v02c01*m02c02) / d : 0);
  float g2 = (d != 0 ? (v01c01*m02c02 - v01c02*m01c01) / d : 0);
  float nx0 = ns[3*v0], ny0 = ns[3*v0+1], nz0 = ns[3*v0+2];
  float nx1 = ns[3*v1], ny1 = ns[3*v1+1], nz1 = ns[3*v1+2];
  float nx2 = ns[3*v2], ny2 = ns[3*v2+1], nz2 = ns[3*v2+2];
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

  /*
  std::cerr << "plane1 " << (x-(mx01+x0))*cx01 + (y-(my01+y0))*cy01 + (z-(mz01+z0))*cz01 << std::endl;
  std::cerr << "plane2 " << (x-(mx02+x0))*cx02 + (y-(my02+y0))*cy02 + (z-(mz02+z0))*cz02 << std::endl;
  std::cerr << "plane3 " << (x-(mx12+x0))*cx12 + (y-(my12+y0))*cy12 + (z-(mz12+z0))*cz12 << std::endl;
  */
}

inline void cut_to_vertex(int v0, int v1, int v2, int a1, int a2,
			  int v10, int v12, int v21, int v20,
			  Vertices &vs, Normals &ns, Triangles &ts, Atoms &v2as)
{
  // Copy edge point.
  int v012 = vs.size()/3, v021 = v012 + 1;
  float x0 = vs[3*v0], y0 = vs[3*v0+1], z0 = vs[3*v0+2];
  add_vertex(vs, x0, y0, z0);
  add_vertex(vs, x0, y0, z0);
  float nx0 = ns[3*v0], ny0 = ns[3*v0+1], nz0 = ns[3*v0+2];
  add_normal(ns, nx0, ny0, nz0);
  add_normal(ns, nx0, ny0, nz0);
  v2as.push_back(a1);
  v2as.push_back(a2);

  add_triangle(ts, v012, v10, v12);
  add_triangle(ts, v10, v1, v12);
  add_triangle(ts, v021, v21, v20);
  add_triangle(ts, v2, v20, v21);
}

inline void cut_to_edge(int v0, int v1, int v2, int a2,
			int v01, int v10, int v12, int v21, int v20, int v02,
			Vertices &vs, Normals &ns, Triangles &ts, Atoms &v2as)
{
  // Copy edge point.
  int v012 = vs.size()/3;
  add_vertex(vs, vs[3*v01], vs[3*v01+1], vs[3*v01+2]);
  add_normal(ns, ns[3*v01],  ns[3*v01+1], ns[3*v01+2]);
  v2as.push_back(a2);

  add_triangle(ts, v0, v01, v02);
  add_triangle(ts, v1, v12, v10);
  add_triangle(ts, v2, v20, v012);
  add_triangle(ts, v2, v012, v21);
}

inline void cut_to_middle(float x, float y, float z, float nx, float ny, float nz,
			  int v0, int v1, int v2, int a0, int a1, int a2,
			  int v01, int v10, int v12, int v21, int v20, int v02,
			  Vertices &vs, Normals &ns, Triangles &ts, Atoms &v2as)
{
  // Add 3 copies of point and normal to middle of triangle.
  int vn = vs.size()/3;
  int vc0 = vn, vc1 = vn+1, vc2 = vn+2;
  for (int c = 0 ; c < 3 ; ++c, ++vn)
    {
      add_vertex(vs, x, y, z);
      add_normal(ns, nx, ny, nz);
    }
  v2as.push_back(a0);
  v2as.push_back(a1);
  v2as.push_back(a2);

  // Add 6 triangles to subdivide this one.
  add_triangle(ts, v0,v01,vc0);
  add_triangle(ts, v0,vc0,v02);
  add_triangle(ts, v1,vc1,v10);
  add_triangle(ts, v1,v12,vc1);
  add_triangle(ts, v2,vc2,v21);
  add_triangle(ts, v2,v20,vc2);
}

// Each triangle vertex is closest to a different atom, so the triangle is to be cut into
// 3 regions using 3 cut lines.  It can happen that the intersection of the 3 lines lies
// outside the triangle in which case the triangle is only cut by two lines, a case handled
// by double_cut_triangle().
inline void cut_triangle_3_lines(int v0, int v1, int v2, int a0, int a1, int a2,
				 float *aa, int as0, int as1,
				 Vertices &vs, Normals &ns,
				 Triangles &ts, Atoms &v2as,
				 Edge_Map &edge_splits)
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
  compute_triple_point(v0, v1, v2, a0, a1, a2, v01, v02, vs, ns, aa, as0, as1,
		       &f1, &f2, &x, &y, &z, &nx, &ny, &nz);

  // Check if triple point is inside triangle.
  float f12 = f1+f2;
  if (f1 > 0 && f2 > 0 && f12 < 1)
    {
      // Point inside triangle.  Divide into 6 new triangles.
      cut_to_middle(x, y, z, nx, ny, nz, v0, v1, v2, a0, a1, a2,
		    v01, v10, v12, v21, v20, v02, vs, ns, ts, v2as);
    }
  else
    {
      // Triple point lies outside triangle.
      int nout = (f1 > 0 ? 0 : 1) + (f2 > 0 ? 0 : 1) + (f12 < 1 ? 0 : 1);
      if (nout == 1)
	{
	  // Project point to triangle edge.
	  if (f1 <= 0)
	    cut_to_edge(v2, v0, v1, a1, v20, v02, v01, v10, v12, v21, vs, ns, ts, v2as);
	  else if (f2 <= 0)
	    cut_to_edge(v0, v1, v2, a2, v01, v10, v12, v21, v20, v02, vs, ns, ts, v2as);
	  else
	    cut_to_edge(v1, v2, v0, a0, v12, v21, v20, v02, v01, v10, vs, ns, ts, v2as);
	}
      else if (nout == 2)
	{
	  // Project point to triangle vertex.
	  if (f1 > 0)
	    cut_to_vertex(v1, v2, v0, a2, a0, v21, v20, v02, v01, vs, ns, ts, v2as);
	  else if (f2 > 0)
	    cut_to_vertex(v2, v0, v1, a0, a1, v02, v01, v10, v12, vs, ns, ts, v2as);
	  else
	    cut_to_vertex(v0, v1, v2, a1, a2, v10, v12, v21, v20, vs, ns, ts, v2as);
	}
    }
}

inline float vertex_atom_distance(int v, int a, Vertices &vs, float *aa, int as0, int as1)
{
  float dx = vs[3*v]-aa[as0*a], dy = vs[3*v+1]-aa[as0*a+as1], dz = vs[3*v+2]-aa[as0*a+2*as1];
  return dx*dx + dy*dy + dz*dz;
}

/*
 * This is only used for iterative subdivision.

static void fix_closest_atoms(Vertices &vs, Triangles &ts, 
			      Atoms &v2as, float *aa, int as0, int as1)
{
  int c = 0;
  int nt = ts.size()/3;
  while (true)
    {
      for (int t = 0 ; t < nt ; ++t)
	{
	  int v0 = ts[3*t], v1 = ts[3*t+1], v2 = ts[3*t+2];
	  int a0 = v2as[v0], a1 = v2as[v1], a2 = v2as[v2];
	  float d0 = vertex_atom_distance(v0,a0,vs,aa,as0,as1);
	  if (a1 != a0 && vertex_atom_distance(v0,a1,vs,aa,as0,as1) < d0)
	    { v2as[v0] = a1; c += 1; }
	  else if (a2 != a0 && vertex_atom_distance(v0,a2,vs,aa,as0,as1) < d0)
	    { v2as[v0] = a2; c += 1; }
	  float d1 = vertex_atom_distance(v1,a1,vs,aa,as0,as1);
	  if (a0 != a1 && vertex_atom_distance(v1,a0,vs,aa,as0,as1) < d1)
	    { v2as[v1] = a0; c += 1; }
	  else if (a2 != a1 && vertex_atom_distance(v1,a2,vs,aa,as0,as1) < d1)
	    { v2as[v1] = a2; c += 1; }
	  float d2 = vertex_atom_distance(v2,a2,vs,aa,as0,as1);
	  if (a0 != a2 && vertex_atom_distance(v2,a0,vs,aa,as0,as1) < d2)
	    { v2as[v2] = a0; c += 1; }
	  else if (a1 != a2 && vertex_atom_distance(v2,a1,vs,aa,as0,as1) < d2)
	    { v2as[v2] = a1; c += 1; }
	}
      if (c == 0)
	break;
      std::cerr << "Adjusted " << c << " closest atoms" << std::endl;
      c = 0;
    }
}
*/

static void fill_vertex_vectors(const FArray &v, const FArray &n, const IArray &v2a,
				Vertices &vs, Normals &ns, Atoms &v2as)
{
  // Get pointers and strides for arrays.
  float *va = v.values(), *na = n.values();
  int *v2aa = v2a.values();
  long vs0 = v.stride(0), vs1 = v.stride(1);
  long ns0 = n.stride(0), ns1 = n.stride(1);
  long v2as0 = v2a.stride(0);

  // Copy vertices and normals to vectors.
  int nv = v.size(0);
  for (int i = 0 ; i < nv ; ++i)
    {
      add_vertex(vs, va[i*vs0], va[i*vs0+vs1], va[i*vs0+2*vs1]);
      add_normal(ns, na[i*ns0], na[i*ns0+ns1], na[i*ns0+2*ns1]);
      v2as.push_back(v2aa[i*v2as0]);
    }
}

static void duplicate_edge_vertices(Vertices &vs, Normals &ns, Atoms &v2as, Edge_Map &edge_splits)
{
  // Duplicate split points.
  Edge_Map edup;
  int vn = vs.size()/3;
  for (Edge_Map::iterator ei = edge_splits.begin() ; ei != edge_splits.end() ; ++ei)
    {
      Edge e = ei->first;
      int ev = ei->second;
      edup[Edge(e.second,e.first)] = vn;
      add_vertex(vs, vs[3*ev], vs[3*ev+1], vs[3*ev+2]);
      add_normal(ns, ns[3*ev], ns[3*ev+1], ns[3*ev+2]);
      v2as.push_back(v2as[e.second]);
      v2as[ev] = v2as[e.first];		// Update atom assignment changed in case vertex 1 assignment changed.
      vn += 1;
    }
  edge_splits.insert(edup.begin(), edup.end());
}

inline bool three_patch_edge(Vertices &vs, int v1, int v2, int a1, int a0,
			     float *aa, long as0, long as1, float *ra, long rs0,
			     Edge_Map &edge_splits, Atoms &v2as, Edge_Map &edge_3p)
{
  float a0x = aa[as0*a0], a0y = aa[as0*a0+as1], a0z = aa[as0*a0+2*as1];
  float a1x = aa[as0*a1], a1y = aa[as0*a1+as1], a1z = aa[as0*a1+2*as1];
  Edge e(min(v1,v2),max(v1,v2));
  int v12 = edge_vertex(edge_splits,e.first,e.second);
  float v12x = vs[3*v12], v12y = vs[3*v12+1], v12z = vs[3*v12+2];
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
      v2as[v12] = a0;
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

inline void split_triangle_3_edges(int v0, int v1, int v2, int v01, int v12, int v20,
				   Triangles &t)
{
  add_triangle(t, v0, v01, v20);
  add_triangle(t, v1, v12, v01);
  add_triangle(t, v2, v20, v12);
  add_triangle(t, v01, v12, v20);
}

inline void split_triangle(int v0, int v1, int v2, int v01, int v12, int v20,
			   Triangles &t)
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

inline int minimize_atom_distance(int v0, int v1, Vertices &vs, int a0, int a1,
				  float *aa, long as0, long as1, float *ra, long rs0,
				  Atoms &v2as)
{
  if (a0 == a1)
    return 0;

  float a0x = aa[as0*a0], a0y = aa[as0*a0+as1], a0z = aa[as0*a0+2*as1];
  float a1x = aa[as0*a1], a1y = aa[as0*a1+as1], a1z = aa[as0*a1+2*as1];
  float v0x = vs[3*v0], v0y = vs[3*v0+1], v0z = vs[3*v0+2];
  float v1x = vs[3*v1], v1y = vs[3*v1+1], v1z = vs[3*v1+2];

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
	{ v2as[v1] = a0; change += 1; }
      if (d01*r0*r0 < d00*r1*r1)
	{ v2as[v0] = a1; change += 1; }
    }
   else
    {
      if (d10 < d11)
	{ v2as[v1] = a0; change += 1; }
      if (d01 < d00)
	{ v2as[v0] = a1; change += 1; }
    }

  float x, y, z, f;
  split_point(v0, v1, v2as[v0], v2as[v1], aa, as0, as1, ra, rs0, vs, false, &x, &y, &z, &f);
  //  if (f <= -0.01 || f >= 1.01)
  /*
  if (f <= 0 || f >= 1)
    {
    std::cerr << "minimize_atom_distance(): inconsitent with split " << f << " edge " << v0 << " " << v1 << " atoms " << a0 << " " << a1 << std::endl;
    std::cerr << "edge vector " << vs[3*v0]-vs[3*v1] << " " << vs[3*v0+1]-vs[3*v1+1] << " " << vs[3*v0+2]-vs[3*v1+2] << std::endl;
    }
  */
  return change;
}

static int minimize_atom_distances(Vertices &vs, Atoms &v2as, const
				   FArray &a, const FArray &r, Triangles &t)
{
  int nt = t.size()/3;
  float *aa = a.values();
  long as0 = a.stride(0), as1 = a.stride(1);
  float *ra = (r.dimension() == 1 ? r.values() : NULL);
  long rs0 = (r.dimension() == 1 ? r.stride(0) : 0);
  int change = 0;
  for (long ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = t[3*ti], v1 = t[3*ti+1], v2 = t[3*ti+2];
      int a0 = v2as[v0], a1 = v2as[v1], a2 = v2as[v2];
      change += minimize_atom_distance(v0, v1, vs, a0, a1, aa, as0, as1, ra, rs0, v2as);
      change += minimize_atom_distance(v1, v2, vs, a1, a2, aa, as0, as1, ra, rs0, v2as);
      change += minimize_atom_distance(v2, v0, vs, a2, a0, aa, as0, as1, ra, rs0, v2as);
    }
  return change;
}

static int divide_long_edges(Vertices &vs, Normals &ns, Triangles &triangles,
			     Atoms &v2as, const FArray &a, const FArray &r,
			     Edge_Map &edge_splits)
{
  int nt = triangles.size()/3;
  float *aa = a.values();
  long as0 = a.stride(0), as1 = a.stride(1);
  float *ra = (r.dimension() == 1 ? r.values() : NULL);
  long rs0 = (r.dimension() == 1 ? r.stride(0) : 0);
  Edge_Map edge_3p;
  for (long ti = 0 ; ti < nt ; ++ti)
     {
      int v0 = triangles[3*ti], v1 = triangles[3*ti+1], v2 = triangles[3*ti+2];
      int a0 = v2as[v0], a1 = v2as[v1], a2 = v2as[v2];
      if (a0 != a1 && a1 != a2 && a2 != a0)
	{
	  three_patch_edge(vs, v0, v1, a0, a2, aa, as0, as1, ra, rs0, edge_splits, v2as, edge_3p);
	  three_patch_edge(vs, v1, v2, a1, a0, aa, as0, as1, ra, rs0, edge_splits, v2as, edge_3p);
	  three_patch_edge(vs, v2, v0, a2, a1, aa, as0, as1, ra, rs0, edge_splits, v2as, edge_3p);
	}
     }
  //  std::cerr << "Three patch edges: " << edge_3p.size() << std::endl;

  Triangles tsplit, tunsplit;
  Edge_Map::iterator ei;
  for (long ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = triangles[3*ti], v1 = triangles[3*ti+1], v2 = triangles[3*ti+2];
      int a0 = v2as[v0], a1 = v2as[v1], a2 = v2as[v2];
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
  //  std::cerr << "Triangle splits produced " << tsplit.size()/3 << " triangles\n";

  // Remove edge split points that are now vertices of triangles.
  for (Edge_Map::iterator ei = edge_3p.begin() ; ei != edge_3p.end() ; ++ei)
    edge_splits.erase(ei->first);
  edge_3p.clear();

  for (int i = 0 ; i < 10 ; ++i)
    {
      int change = minimize_atom_distances(vs, v2as, a, r, tsplit);
      change += minimize_atom_distances(vs, v2as, a, r, tunsplit);
      if (change == 0)
	break;
      //      std::cerr << "Fixed " << change << " atom assignments\n";
    }

  compute_edge_split_points(vs, ns, tsplit, v2as, a, r, edge_splits);
  // The unsplit can also have edges that need split because vertex was assigned a new atom.
  compute_edge_split_points(vs, ns, tunsplit, v2as, a, r, edge_splits);

  triangles.clear();
  triangles.insert(triangles.end(), tunsplit.begin(), tunsplit.end());
  triangles.insert(triangles.end(), tsplit.begin(), tsplit.end());

  return tsplit.size()/3;
}

static void divide_triangles(Vertices &vs, Normals &ns, Triangles &triangles,
			     Atoms &v2as, const FArray &a,
			     Edge_Map &edge_splits, Triangles &ts)
{
  duplicate_edge_vertices(vs, ns, v2as, edge_splits);

  int nt = triangles.size()/3;
  float *aa = a.values();
  long as0 = a.stride(0), as1 = a.stride(1);
  for (long ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = triangles[3*ti], v1 = triangles[3*ti+1], v2 = triangles[3*ti+2];
      int a0 = v2as[v0], a1 = v2as[v1], a2 = v2as[v2];
      if (a0 == a1 && a1 == a2)
	add_triangle(ts, v0, v1, v2);  // copy triangle, no subdivision
      else if (a0 != a1 && a1 != a2 && a2 != a0)
	cut_triangle_3_lines(v0, v1, v2, a0, a1, a2, aa, as0, as1,
			     vs, ns, ts, v2as, edge_splits);
      else
	// Cut triangle along one line.
	cut_triangle_1_line(v0, v1, v2, a0, a1, a2, ts, edge_splits);
    }
}

static void copy_triangles(const IArray &t, Triangles &triangles)
{
  int nt = t.size(0);
  int *ta = t.values();
  long ts0 = t.stride(0), ts1 = t.stride(1);
  for (int i = 0 ; i < nt ; ++i)
    add_triangle(triangles, ta[ts0*i], ta[ts0*i+ts1], ta[ts0*i+2*ts1]);
}

static void sharp_patches(const FArray &v, const FArray &n, const IArray &t,
			  const IArray &v2a, const FArray &a, const FArray &r,
			  int refinement_steps,
			  Vertices &vs, Normals &ns,
			  Triangles &ts, Atoms &v2as)
{
  // Convert vertex geometry to vectors.
  fill_vertex_vectors(v, n, v2a, vs, ns, v2as);

  Triangles triangles;
  copy_triangles(t, triangles);

  // Make vertices for split edges.
  Edge_Map edge_splits;
  compute_edge_split_points(vs, ns, triangles, v2as, a, r, edge_splits);

  // Split edges that span three atom zones for better boundaries
  // for narrow strip atom zones.
  for (int i = 0 ; i < refinement_steps ; ++i)
    if (divide_long_edges(vs, ns, triangles, v2as, a, r, edge_splits) == 0)
      break;

   // Make subdivided triangles along atom zone boundaries.
  divide_triangles(vs, ns, triangles, v2as, a, edge_splits, ts);

  //  fix_closest_atoms(*vs, *ts, *v2as, aa, as0, as1);
  //  std::cerr << "vertices " << nv << " - " << vs->size()/3 << " tri " << nt << " - " << ts->size()/3 << std::endl;
}
			  
// ----------------------------------------------------------------------------
//
extern "C" PyObject *sharp_edge_patches(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray vertices, normals, axyz, radii;
  IArray triangles, v2a;
  int refinement_steps = 0;
  const char *kwlist[] = {"vertices", "normals", "triangles", "v2a", "atom_xyz", "atom_radii",
			  "refinement_steps", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&O&|O&i"), (char **)kwlist,
				   parse_float_n3_array, &vertices,
				   parse_float_n3_array, &normals,
				   parse_int_n3_array, &triangles,
				   parse_int_n_array, &v2a,
				   parse_float_n3_array, &axyz,
				   parse_float_n_array, &radii,
				   &refinement_steps))
    return NULL;

  if (normals.size(0) != vertices.size(0))
    {
      PyErr_SetString(PyExc_ValueError, "normals and vertices have different lengths");
      return NULL;
    }
  if (v2a.size(0) != vertices.size(0))
    {
      PyErr_SetString(PyExc_ValueError, "vertex map and vertices have different sizes");
      return NULL;
    }
  if (radii.dimension() == 1 && radii.size(0) != axyz.size(0))
    {
      PyErr_SetString(PyExc_ValueError, "atom coordinates and radii arrays have different sizes");
      return NULL;
    }

  Vertices vs;
  Normals ns;
  Triangles ts;
  Atoms v2as;
  Py_BEGIN_ALLOW_THREADS
    sharp_patches(vertices, normals, triangles, v2a, axyz, radii, refinement_steps,
		  vs, ns, ts, v2as);
  Py_END_ALLOW_THREADS

  int nv = vs.size()/3, nt = ts.size()/3;
  PyObject *vsa = c_array_to_python(vs, nv, 3);
  PyObject *nsa = c_array_to_python(ns, nv, 3);
  PyObject *tsa = c_array_to_python(ts, nt, 3);
  PyObject *v2asa = c_array_to_python(v2as);
  PyObject *r = python_tuple(vsa, nsa, tsa, v2asa);

  for (long ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = ts[3*ti], v1 = ts[3*ti+1], v2 = ts[3*ti+2];
      int a0 = v2as[v0], a1 = v2as[v1], a2 = v2as[v2];
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
//
static void unique_vertices(const FArray &vertices, int *vmap)
{
  std::map<Vertex,int> vm;
  int nv = vertices.size(0);
  long vs0 = vertices.stride(0), vs1 = vertices.stride(1);
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
