#include <map>		// use std::map
#include <iostream>

#include "pythonarray.h"		// use python_float_array
#include "rcarray.h"			// use FArray, IArray

typedef std::pair<int,int> Edge;
typedef std::map<Edge,int> Edge_Map;

#define min(a,b) (a<b ? a : b)
#define max(a,b) (a>b ? a : b)

static void count_edge_splits(const IArray &t, const IArray &v2a,
			      Edge_Map &edge_splits, int *ts2, int *ts3)
{
  int nt = t.size(0);
  int *ta = t.values(), *v2aa = v2a.values();
  long ts0 = t.stride(0), ts1 = t.stride(1), v2as0 = v2a.stride(0);
  int vc = v2a.size(0);
  int t2 = 0, t3 = 0;
  for (int ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = ta[ts0*ti], v1 = ta[ts0*ti+ts1], v2 = ta[ts0*ti+2*ts1];
      int a0 = v2aa[v2as0*v0], a1 = v2aa[v2as0*v1], a2 = v2aa[v2as0*v2];
      int s = 0;
      if (a0 != a1)
	{
	  Edge e(min(v0,v1),max(v0,v1));
	  if (edge_splits.find(e) == edge_splits.end())
	    { edge_splits[e] = vc; vc += 2; }
	  s += 1;
	}
      if (a1 != a2)
	{
	  Edge e(min(v1,v2),max(v1,v2));
	  if (edge_splits.find(e) == edge_splits.end())
	    { edge_splits[e] = vc; vc += 2; }
	  s += 1;
	}
      if (a2 != a0)
	{
	  Edge e(min(v2,v0),max(v2,v0));
	  if (edge_splits.find(e) == edge_splits.end())
	    { edge_splits[e] = vc; vc += 2; }
	  s += 1;
	}
      if (s == 2)
	t2 += 1;
      else if (s == 3)
	t3 += 1;
    }
  *ts2 = t2;
  *ts3 = t3;
}

inline void add_vertex(std::vector<float> *v, float x, float y, float z)
{
  v->push_back(x);
  v->push_back(y);
  v->push_back(z);
}

inline void add_triangle(std::vector<int> *t, int v0, int v1, int v2)
{
  t->push_back(v0);
  t->push_back(v1);
  t->push_back(v2);
}
inline void bisect_triangle(int v0, int v1, int v2, int a0, int a1, int a2,
			    std::vector<int> *ts, Edge_Map &edge_splits)
{
  // 2 edges split
  // First put in standard orientation with edges 02 and 12 split.
  if (a1 == a2)
    { int temp = v0; v0 = v1; v1 = v2; v2 = temp; }
  else if (a2 == a0)
    { int temp = v2; v2 = v1; v1 = v0; v0 = temp; }

  // Add 3 triangles to subdivide this one
  int v12, v21, v20, v02;
  if (v1 < v2) { v12 = edge_splits[Edge(v1,v2)]; v21 = v12+1; }
  else { v21 = edge_splits[Edge(v2,v1)]; v12 = v21+1; }
  if (v2 < v0) { v20 = edge_splits[Edge(v2,v0)]; v02 = v20+1; }
  else { v02 = edge_splits[Edge(v0,v2)]; v20 = v02+1; }

  add_triangle(ts, v0,v1,v12);
  add_triangle(ts, v0,v12,v02);
  add_triangle(ts, v2,v20,v21);
}

inline int split_edge(int v0, int v1, int a0, int a1, float *va, int vs0, int vs1,
		      float *na, int ns0, int ns1, float *aa, int as0, int as1,
		      std::vector<float> *vs, std::vector<float> *ns, std::vector<int> *v2as)
{
  // Find position to split along edge.
  // Equidistant from atoms: f = (0.5*(a1xyz+a2xyz)-v0xyz, a1xyz-a0xyz) / (v1xyz-v0xyz, a1xyz-a0xyz)
  float x0 = va[vs0*v0], y0 = va[vs0*v0+vs1], z0 = va[vs0*v0+2*vs1];
  float x1 = va[vs0*v1], y1 = va[vs0*v1+vs1], z1 = va[vs0*v1+2*vs1];
  float dx = x1-x0, dy = y1-y0, dz = z1-z0;
  float ax0 = aa[as0*a0], ay0 = aa[as0*a0+as1], az0 = aa[as0*a0+2*as1];
  float ax1 = aa[as0*a1], ay1 = aa[as0*a1+as1], az1 = aa[as0*a1+2*as1];
  float dax = ax1-ax0, day = ay1-ay0, daz = az1-az0;
  float cx = 0.5*(ax0+ax1)-x0, cy = 0.5*(ay0+ay1)-y0, cz = 0.5*(az0+az1)-z0;
  float dvda = dx*dax + dy*day + dz*daz;
  float dcda = cx*dax + cy*day + cz*daz;
  float f1 = (dvda != 0 ? dcda / dvda : 0.5);
  float f0 = 1-f1;
  if (f1 < 0 || f1 > 1)
    std::cerr << "split edge " << v0 << " " << v1 << " " << a0 << " " << a1 << " " << f1 << std::endl;
  float x = f0*x0 + f1*x1, y = f0*y0 + f1*y1, z = f0*z0 + f1*z1;
  //      std::cerr << "split point " << v0 << " " << v1 << " " << x << " " << y << " " << z << std::endl;

  // Make 2 copies of vertex at split position
  add_vertex(vs, x, y, z);
  add_vertex(vs, x, y, z);

  // Make 2 copies of normal at split position
  float nx = f0*na[ns0*v0] + f1*na[ns0*v1];
  float ny = f0*na[ns0*v0+ns1] + f1*na[ns0*v1+ns1];
  float nz = f0*na[ns0*v0+2*ns1] + f1*na[ns0*v1+2*ns1];
  float n2 = sqrt(nx*nx + ny*ny + nz*nz);
  if (n2 > 0)
	{ nx /= n2; ny /= n2 ; nz /= n2; }
  add_vertex(ns, nx, ny, nz);
  add_vertex(ns, nx, ny, nz);

  // Assign atom index for two new vertices.
  v2as->push_back(a0);
  v2as->push_back(a1);

  return v2as->size() - 2;
}

inline void split_edges(std::vector<float> *vs, std::vector<float> *ns, std::vector<int> *v2as,
			float *va, int vs0, int vs1, float *na, int ns0, int ns1,
			int *v2aa, int v2as0, float *aa, int as0, int as1,
			Edge_Map &edge_splits)
{
  for (Edge_Map::iterator ei = edge_splits.begin() ; ei != edge_splits.end() ; ++ei)
    {
      const Edge &e = ei->first;
      int v0 = e.first, v1 = e.second;
      int a0 = v2aa[v2as0*v0], a1 = v2aa[v2as0*v1];
      ei->second = split_edge(v0, v1, a0, a1, va, vs0, vs1, na, ns0, ns1, aa, as0, as1, vs, ns, v2as);
    }
}


inline bool double_bisect_triangle(int v0, int v1, int v2, int a0, int a1, int a2,
				   int v01, int v10, int v12, int v21, int v20, int v02,
				   float *va, int vs0, int vs1, float *na, int ns0, int ns1, 
				   float *aa, int as0, int as1,
				   std::vector<float> *vs, std::vector<float> *ns,
				   std::vector<int> *ts, std::vector<int> *v2as)
{
  float ex0 = (*vs)[vs0*v12], ey0 = (*vs)[vs0*v12+vs1], ez0 = (*vs)[vs0*v12+2*vs1];

  float ax0 = aa[as0*a0], ay0 = aa[as0*a0+as1], az0 = aa[as0*a0+2*as1];
  float ax1 = aa[as0*a1], ay1 = aa[as0*a1+as1], az1 = aa[as0*a1+2*as1];
  //  float ax2 = aa[as0*a2], ay2 = aa[as0*a2+as1], az2 = aa[as0*a2+2*as1];

  float dx0 = ex0-ax0, dy0 = ey0-ay0, dz0 = ez0-az0;
  float dx1 = ex0-ax1, dy1 = ey0-ay1, dz1 = ez0-az1;
  float d0 = dx0*dx0 + dy0*dy0 + dz0*dz0;
  float d1 = dx1*dx1 + dy1*dy1 + dz1*dz1;
  if (d0 >= d1)
    return false;

  // Double bisect edge 12.
  int v12a = split_edge(v1, v2, a1, a0, va, vs0, vs1, na, ns0, ns1, aa, as0, as1, vs, ns, v2as);
  int v21a = v12a + 1;
  int v12b = split_edge(v1, v2, a0, a2, va, vs0, vs1, na, ns0, ns1, aa, as0, as1, vs, ns, v2as);
  int v21b = v12b + 1;
  add_triangle(ts, v1, v12a, v10);
  add_triangle(ts, v2, v20, v21b);
  add_triangle(ts, v0, v01, v21a);
  add_triangle(ts, v0, v21a, v12b);
  add_triangle(ts, v0, v12b, v02);

  /*
    For iterative subdivision.  This requires used of fix_closest_atoms().
  (*v2as)[v21] = a0;
  (*v2as)[v12] = a0;
  */

  //  std::cerr << "double bisect " << v1 << " " << v2 << " " << a1 << " " << a2 << " " << a0 << std::endl;

  return true;
}

inline void trisect_triangle(int v0, int v1, int v2, int a0, int a1, int a2,
			     float *va, int vs0, int vs1, float *na, int ns0, int ns1, 
			     float *aa, int as0, int as1,
			     std::vector<float> *vs, std::vector<float> *ns,
			     std::vector<int> *ts, std::vector<int> *v2as, Edge_Map &edge_splits)
{
  // All 3 edges split
  int v01, v10, v12, v21, v20, v02;
  if (v0 < v1) { v01 = edge_splits[Edge(v0,v1)]; v10 = v01+1; }
  else { v10 = edge_splits[Edge(v1,v0)]; v01 = v10+1; }
  if (v1 < v2) { v12 = edge_splits[Edge(v1,v2)]; v21 = v12+1; }
  else { v21 = edge_splits[Edge(v2,v1)]; v12 = v21+1; }
  if (v2 < v0) { v20 = edge_splits[Edge(v2,v0)]; v02 = v20+1; }
  else { v02 = edge_splits[Edge(v0,v2)]; v20 = v02+1; }

  if (double_bisect_triangle(v0, v1, v2, a0, a1, a2, v01, v10, v12, v21, v20, v02,
			     va, vs0, vs1, na, ns0, ns1, aa, as0, as1, vs, ns, ts, v2as) ||
      double_bisect_triangle(v1, v2, v0, a1, a2, a0, v12, v21, v20, v02, v01, v10,
			     va, vs0, vs1, na, ns0, ns1, aa, as0, as1, vs, ns, ts, v2as) ||
      double_bisect_triangle(v2, v0, v1, a2, a0, a1, v20, v02, v01, v10, v12, v21,
			     va, vs0, vs1, na, ns0, ns1, aa, as0, as1, vs, ns, ts, v2as))
    return;

  // Add mid-triangle vertex (3 copies).
  float x0 = va[vs0*v0], y0 = va[vs0*v0+vs1], z0 = va[vs0*v0+2*vs1];
  float x1 = va[vs0*v1], y1 = va[vs0*v1+vs1], z1 = va[vs0*v1+2*vs1];
  float x2 = va[vs0*v2], y2 = va[vs0*v2+vs1], z2 = va[vs0*v2+2*vs1];
  float x01 = x1-x0, y01 = y1-y0, z01 = z1-z0;
  float x02 = x2-x0, y02 = y2-y0, z02 = z2-z0;

  float ax0 = aa[as0*a0], ay0 = aa[as0*a0+as1], az0 = aa[as0*a0+2*as1];
  float ax1 = aa[as0*a1], ay1 = aa[as0*a1+as1], az1 = aa[as0*a1+2*as1];
  float ax2 = aa[as0*a2], ay2 = aa[as0*a2+as1], az2 = aa[as0*a2+2*as1];
  float cx01 = ax1-ax0, cy01 = ay1-ay0, cz01 = az1-az0;
  float cx02 = ax2-ax0, cy02 = ay2-ay0, cz02 = az2-az0;

  //	  float cx12 = ax2-ax1, cy12 = ay2-ay1, cz12 = az2-az1;
  float mx01 = 0.5*(ax0+ax1)-x0, my01 = 0.5*(ay0+ay1)-y0, mz01 = 0.5*(az0+az1)-z0;
  float mx02 = 0.5*(ax0+ax2)-x0, my02 = 0.5*(ay0+ay2)-y0, mz02 = 0.5*(az0+az2)-z0;
  //	  float mx12 = 0.5*(ax1+ax2)-x0, my12 = 0.5*(ay1+ay2)-y0, mz12 = 0.5*(az1+az2)-z0;

  float v01c01 = x01*cx01 + y01*cy01 + z01*cz01;
  float v01c02 = x01*cx02 + y01*cy02 + z01*cz02;
  float v02c01 = x02*cx01 + y02*cy01 + z02*cz01;
  float v02c02 = x02*cx02 + y02*cy02 + z02*cz02;
  float m01c01 = mx01*cx01 + my01*cy01 + mz01*cz01;
  float m02c02 = mx02*cx02 + my02*cy02 + mz02*cz02;

  float d = v01c01*v02c02 - v02c01*v01c02;
  float f1 = (d != 0 ? (v02c02*m01c01 - v02c01*m02c02) / d : 0);
  float f2 = (d != 0 ? (v01c01*m02c02 - v01c02*m01c01) / d : 0);
  //  if (f1 < 0 || f1 > 1 || f2 < 0 || f2 > 1) tpo += 1; else tpi += 1;
  if (f1 < 0) f1 = 0; else if (f1 > 1) f1 = 1;
  if (f2 < 0) f2 = 0; else if (f2 > 1) f2 = 1;
  float f12 = f1 + f2;
  if (f12 > 1) { f1 /= f12; f2 /= f12; }
  //	  std::cerr << "tri center " << f1 << " " << f2 << std::endl;
  float x = x0 + f1*x01 + f2*x02, y = y0 + f1*y01 + f2*y02, z = z0 + f1*z01 + f2*z02;

  /*
  std::cerr << "plane1 " << (x-(mx01+x0))*cx01 + (y-(my01+y0))*cy01 + (z-(mz01+z0))*cz01 << std::endl;
  std::cerr << "plane2 " << (x-(mx02+x0))*cx02 + (y-(my02+y0))*cy02 + (z-(mz02+z0))*cz02 << std::endl;
  std::cerr << "plane3 " << (x-(mx12+x0))*cx12 + (y-(my12+y0))*cy12 + (z-(mz12+z0))*cz12 << std::endl;
  */

  //	  float x = (x0+x1+x2)/3, y = (y0+y1+y2)/3, z = (z0+z1+z2)/3;
  //	  std::cerr << "tri center " << ti << " " << x << " " << y << " " << z << std::endl;

  // Add mid-triangle normal (3 copies).
  float nx0 = na[ns0*v0], ny0 = na[ns0*v0+ns1], nz0 = na[ns0*v0+2*ns1];
  float nx1 = na[ns0*v1], ny1 = na[ns0*v1+ns1], nz1 = na[ns0*v1+2*ns1];
  float nx2 = na[ns0*v2], ny2 = na[ns0*v2+ns1], nz2 = na[ns0*v2+2*ns1];
  float nx01 = nx1-nx0, ny01 = ny1-ny0, nz01 = nz1-nz0;
  float nx02 = nx2-nx0, ny02 = ny2-ny0, nz02 = nz2-nz0;
  float nx = nx0 + f1*nx01 + f2*nx02, ny = ny0 + f1*ny01 + f2*ny02, nz = nz0 + f1*nz01 + f2*nz02;
  //	  float nx = (nx0+nx1+nx2)/3, ny = (ny0+ny1+ny2)/3, nz = (nz0+nz1+nz2)/3;
  float n2 = sqrt(nx*nx + ny*ny + nz*nz);
  if (n2 > 0)
    { nx /= n2; ny /= n2 ; nz /= n2; }

  int vn = vs->size()/3;
  int vc0 = vn, vc1 = vn+1, vc2 = vn+2;
  v2as->push_back(a0);
  v2as->push_back(a1);
  v2as->push_back(a2);
  for (int c = 0 ; c < 3 ; ++c, ++vn)
    {
      add_vertex(vs, x, y, z);
      add_vertex(ns, nx, ny, nz);
    }

  // Add 6 triangles to subdivide this one.
  add_triangle(ts, v0,v01,vc0);
  add_triangle(ts, v0,vc0,v02);
  add_triangle(ts, v1,vc1,v10);
  add_triangle(ts, v1,v12,vc1);
  add_triangle(ts, v2,vc2,v21);
  add_triangle(ts, v2,v20,vc2);
}

inline float vertex_atom_distance(int v, int a, std::vector<float> &vs, float *aa, int as0, int as1)
{
  float dx = vs[3*v]-aa[as0*a], dy = vs[3*v+1]-aa[as0*a+as1], dz = vs[3*v+2]-aa[as0*a+2*as1];
  return dx*dx + dy*dy + dz*dz;
}

static void fix_closest_atoms(std::vector<float> &vs, std::vector<int> &ts, 
			      std::vector<int> &v2as, float *aa, int as0, int as1)
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

static void sharp_patches(const FArray &v, const FArray &n, const IArray &t, const IArray &v2a, const FArray &a,
			  std::vector<float> *vs, std::vector<float> *ns,
			  std::vector<int> *ts, std::vector<int> *v2as)
{
  Edge_Map edge_splits;
  int t2, t3;
  count_edge_splits(t, v2a, edge_splits, &t2, &t3);
  std::cerr << "2 edge splits " << t2 << ", 3 edge splits " << t3 << std::endl;

  // Compute geometry sizes with new vertices and triangles.
  int ne = edge_splits.size();
  int nv = v.size(0);
  int nt = t.size(0);

  std::cerr << "old vert " << nv << " tri " << nt << std::endl;

  // Get pointers and strides for original geometry
  float *va = v.values(), *na = n.values(), *aa = a.values();
  int *ta = t.values(), *v2aa = v2a.values();
  long vs0 = v.stride(0), vs1 = v.stride(1);
  long ns0 = n.stride(0), ns1 = n.stride(1);
  long ts0 = t.stride(0), ts1 = t.stride(1);
  long v2as0 = v2a.stride(0), as0 = a.stride(0), as1 = a.stride(1);

  // Copy original vertices and normals.
  for (int i = 0 ; i < nv ; ++i)
    {
      add_vertex(vs, va[i*vs0], va[i*vs0+vs1], va[i*vs0+2*vs1]);
      add_vertex(ns, na[i*ns0], na[i*ns0+ns1], na[i*ns0+2*ns1]);
      v2as->push_back(v2aa[i*v2as0]);
    }

  // Make vertices for edge splits.
  split_edges(vs, ns, v2as, va, vs0, vs1, na, ns0, ns1, v2aa, v2as0, aa, as0, as1, edge_splits);

  // Make subdivided triangles.
  for (long ti = 0 ; ti < nt ; ++ti)
    {
      int v0 = ta[ts0*ti], v1 = ta[ts0*ti+ts1], v2 = ta[ts0*ti+2*ts1];
      int a0 = v2aa[v2as0*v0], a1 = v2aa[v2as0*v1], a2 = v2aa[v2as0*v2];
      if (a0 == a1 && a1 == a2)
	add_triangle(ts, v0, v1, v2);  // copy triangle, no subdivision
      else if (a0 != a1 && a1 != a2 && a2 != a0)
	trisect_triangle(v0, v1, v2, a0, a1, a2, va, vs0, vs1, na, ns0, ns1, aa, as0, as1,
			 vs, ns, ts, v2as, edge_splits);
      else
	bisect_triangle(v0, v1, v2, a0, a1, a2, ts, edge_splits);
    }

  //  fix_closest_atoms(*vs, *ts, *v2as, aa, as0, as1);

  std::cerr << "new vert " << vs->size()/3 << " tri " << ts->size()/3 << std::endl;
}
			  
// ----------------------------------------------------------------------------
//
extern "C" PyObject *sharp_edge_patches(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray vertices, normals, axyz;
  IArray triangles, v2a;
  const char *kwlist[] = {"vertices", "normals", "triangles", "v2a", "atom_xyz", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&O&"), (char **)kwlist,
				   parse_float_n3_array, &vertices,
				   parse_float_n3_array, &normals,
				   parse_int_n3_array, &triangles,
				   parse_int_n_array, &v2a,
				   parse_float_n3_array, &axyz))
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

  std::vector<float> vs, ns;
  std::vector<int> ts, v2as;
  sharp_patches(vertices, normals, triangles, v2a, axyz, &vs, &ns, &ts, &v2as);

  int nv = vs.size()/3, nt = ts.size()/3;
  PyObject *vsa = c_array_to_python(vs, nv, 3);
  PyObject *nsa = c_array_to_python(ns, nv, 3);
  PyObject *tsa = c_array_to_python(ts, nt, 3);
  PyObject *v2asa = c_array_to_python(v2as);
  PyObject *r = python_tuple(vsa, nsa, tsa, v2asa);
  return r;
}
