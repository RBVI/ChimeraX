// ----------------------------------------------------------------------------
// Compute subdivide surface triangulation where each triangle is divided
// into 4 triangles by connecting the midpoints of the triangle edges.
//
// Alot of the code is devoted to making only one copy of each edge mid-points
// in the new vertex array.
//
#include <Python.h>			// use PyObject

//#include <iostream>			// use std::cerr for debugging
#include <set>				// use std::set<>
#include <stdexcept>			// use std::runtime_error

#include <math.h>			// use sqrt()

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray, IArray

typedef std::vector<int> Indices;

// ----------------------------------------------------------------------------
//
class Edge
{
public:
  int vindex1, vindex2;		// vindex1 <= vindex2
  mutable int ev1, ev2;		// Edge interior vertex index range.
				// Need mutable to modify in std::set<Edge>
  bool operator<(const Edge &e) const
  {
    return (vindex1 < e.vindex1)
      || (vindex1 == e.vindex1 && vindex2 < e.vindex2);
  }
  void set_endpoints(int v1, int v2)
  {
    if (v1 < v2)	{ vindex1 = v1; vindex2 = v2; }
    else		{ vindex1 = v2; vindex2 = v1; }
  }
  void edge_vertices(Indices &ev, bool reverse) const
  {
    int n = (ev2 - ev1) + 3;
    ev.resize(n);
    if (reverse)
      {
	ev[0] = vindex2;
	for (int i = 1 ; i < n-1 ; ++i)
	  ev[i] = ev2 - i + 1;
	ev[n-1] = vindex1;
      }
    else
      {
	ev[0] = vindex1;
	for (int i = 1 ; i < n-1 ; ++i)
	  ev[i] = ev1 + i - 1;
	ev[n-1] = vindex2;
      }
  }
};

// ----------------------------------------------------------------------------
//
class Geometry
{
public:
  std::set<Edge> eset;
  std::vector<float> va, na;
  std::vector<int> ta;
};

// ----------------------------------------------------------------------------
//
static void parse_geometry(PyObject *vertex_array, PyObject *triangle_array,
			   PyObject *normals_array, FArray &varray,
			   FArray &narray, IArray &tarray);
static PyObject *python_geometry(PyObject *v, PyObject *t, PyObject *n);
static void find_edges(const IArray &tc, std::set<Edge> &eset);
static void subdivided_vertices(const FArray &varray, std::set<Edge> &eset,
				float *varray2);
static void interpolated_vertex(float *v1, float *v2, float frac, float *vf);
static void subdivided_triangles(const IArray &tc, std::set<Edge> &eset,
				 int *tarray2);
static void normalize_normals(float *na, int nv);
static void point_vector(const FArray &varray, std::vector<float> &va);
static void subdivide_triangle(int i1, int i2, int i3, float elength,
			       Geometry &g);
static void subdivide_edge(int i0, int i1, float elength, Geometry &g,
			   Indices &ev);
static float distance(float *v0, float *v1);
static void triangle_strips(int i1, int i2, int i3, Indices &e2, Indices &e3,
			    float elength, Geometry &g);
static void stitch_strip(Indices &e1, Indices &e2, Geometry &g);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
subdivide_triangles(PyObject *s, PyObject *args, PyObject *keywds)
{
  FArray varray, narray;
  IArray tarray;
  const char *kwlist[] = {"vertices", "triangles", "normals", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&|O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray,
				   parse_float_n3_array, &narray))
    return NULL;
  if (narray.size() > 0 && narray.size(0) != varray.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "subdivide_triangles(): normal array size is not the same as vertex array size");
      return NULL;
    }

  // Record all unique edges, so one midpoint is created just once
  // for edges that occur in 2 or more triangles.
  std::set<Edge> eset;  
  find_edges(tarray, eset);

  // Make new vertex array including mid-points.
  int nv = varray.size(0) + eset.size();
  float *va2;
  PyObject *varray2 = python_float_array(nv, 3, &va2);
  subdivided_vertices(varray, eset, va2);

  // Make new triangle array.
  int *ta2, nt = 4*tarray.size(0);
  PyObject *tarray2 = python_int_array(nt, 3, &ta2);
  subdivided_triangles(tarray, eset, ta2);

  // Make new normals array including mid-points.
  PyObject *narray2 = NULL;
  if (narray.size() > 0)
    {
      float *na2;
      narray2 = python_float_array(nv, 3, &na2);
      subdivided_vertices(narray, eset, na2);
      normalize_normals(na2, nv);
    }

  return python_geometry(varray2, tarray2, narray2);
}

// ----------------------------------------------------------------------------
//
static void parse_geometry(PyObject *vertex_array, PyObject *triangle_array,
			   PyObject *normals_array, FArray &varray,
			   FArray &narray, IArray &tarray)
{
 varray = array_from_python(vertex_array, 2, Numeric_Array::Float);
  if (varray.size(1) != 3)
    throw std::runtime_error("vertex array second dimension must be 3");

  tarray = array_from_python(triangle_array, 2, Numeric_Array::Int);
  if (tarray.size(1) != 3)
    throw std::runtime_error("triangle array second dimension must be 3");

  if (normals_array)
    {
      narray = array_from_python(normals_array, 2, Numeric_Array::Float);
      if (narray.size(1) != 3)
	throw std::runtime_error("normal array second dimension must be 3");
      if (narray.size(0) != varray.size(0))
	throw std::runtime_error("normal array size differs from vertex array");
    }
}

// ----------------------------------------------------------------------------
//
static PyObject *python_geometry(PyObject *v, PyObject *t, PyObject *n)
{
  PyObject *vtn = PyTuple_New(n ? 3 : 2);
  PyTuple_SetItem(vtn, 0, v);
  PyTuple_SetItem(vtn, 1, t);
  if (n)
    PyTuple_SetItem(vtn, 2, n);
  return vtn;
}

// ----------------------------------------------------------------------------
//
static void find_edges(const IArray &triangles, std::set<Edge> &eset)
{
  Edge e;
  int *ta = triangles.values();
  int s0 = triangles.stride(0), s1 = triangles.stride(1);
  int tcount = triangles.size(0);
  for (int t = 0 ; t < tcount ; ++t)
    {
      int v1 = ta[s0*t], v2 = ta[s0*t+s1], v3 = ta[s0*t+2*s1];
      e.set_endpoints(v1, v2);
      eset.insert(e);
      e.set_endpoints(v2, v3);
      eset.insert(e);
      e.set_endpoints(v3, v1);
      eset.insert(e);
    }
}

// ----------------------------------------------------------------------------
//
static void subdivided_vertices(const FArray &varray, std::set<Edge> &eset,
				float *varray2)
{
  FArray vc = varray.contiguous_array();
  float *vfrom = vc.values();
  int vsz = vc.size();
  for (int k = 0 ; k < vsz ; ++k)
    varray2[k] = vfrom[k];			// Copy original vertices.

  // Compute mid-points
  int vi = vc.size(0);
  for (std::set<Edge>::iterator ei = eset.begin() ; ei != eset.end() ; ++ei)
  {
    interpolated_vertex(vfrom+3*(*ei).vindex1, vfrom+3*(*ei).vindex2, 0.5,
			&(varray2[3*vi]));
    (*ei).ev1 = vi;
    vi += 1;
  }
}

// ----------------------------------------------------------------------------
//
static void interpolated_vertex(float *v1, float *v2, float frac, float *vf)
{
  float f1 = 1.0 - frac, f2 = frac;
  for (int a = 0 ; a < 3 ; ++a)
    vf[a] = f1 * v1[a] + f2 * v2[a];
}

// ----------------------------------------------------------------------------
//
static void subdivided_triangles(const IArray &tc, std::set<Edge> &eset,
				 int *tarray2)
{
  int tcount = tc.size(0);
  int *tcv = tc.values();
  Edge e;
  for (int t = 0 ; t < tcount ; ++t)
    {
      int v1 = tcv[3*t], v2 = tcv[3*t+1], v3 = tcv[3*t+2];
      e.set_endpoints(v1, v2);
      int v12 = eset.find(e)->ev1;
      e.set_endpoints(v2, v3);
      int v23 = eset.find(e)->ev1;
      e.set_endpoints(v3, v1);
      int v31 = eset.find(e)->ev1;
      int *ts = &(tarray2[12*t]);
      ts[0] = v1; ts[1] = v12; ts[2] = v31;
      ts[3] = v2; ts[4] = v23; ts[5] = v12;
      ts[6] = v3; ts[7] = v31; ts[8] = v23;
      ts[9] = v12; ts[10] = v23; ts[11] = v31;
    }
}

// ----------------------------------------------------------------------------
//
static void normalize_normals(float *na, int nv)
{
  int nv3 = 3*nv;
  for (int k = 0 ; k < nv3 ; k += 3)
    {
      float nx = na[k], ny = na[k+1], nz = na[k+2];
      float norm = sqrt(nx*nx + ny*ny + nz*nz);
      if (norm > 0)
	{ na[k] = nx/norm; na[k+1] = ny/norm; na[k+2] = nz/norm; }
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
subdivide_mesh(PyObject *s, PyObject *args, PyObject *keywds)
{
  float elength;
  FArray varray, narray;
  IArray tarray;
  const char *kwlist[] = {"vertices", "triangles", "normals", "edge_length",
			  NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&f"),
				   (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray,
				   parse_float_n3_array, &narray,
				   &elength))
    return NULL;
  if (narray.size() > 0 && narray.size(0) != varray.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "subdivide_triangle(): normal array size is not the same as vertex array size");
      return NULL;
    }

  Geometry g;

  point_vector(varray, g.va);
  point_vector(narray, g.na);

  // Loop over triangles and subdivide.
  int nt = tarray.size(0);
  int ts0 = tarray.stride(0), ts1 = tarray.stride(1);
  int *tp = tarray.values();
  for (int k = 0 ; k < nt ; ++k)
    {
      int v0 = tp[ts0*k], v1 = tp[ts0*k + ts1], v2 = tp[ts0*k + 2*ts1];
      subdivide_triangle(v0,v1,v2, elength, g);
    }
  normalize_normals(&g.na[0], g.na.size()/3);

  PyObject *vpy = c_array_to_python(&g.va[0], g.va.size()/3, 3);
  PyObject *npy = c_array_to_python(&g.na[0], g.na.size()/3, 3);
  PyObject *tpy = c_array_to_python(&g.ta[0], g.ta.size()/3, 3);

  return python_geometry(vpy, tpy, npy);
}

// ----------------------------------------------------------------------------
//
static void point_vector(const FArray &varray, std::vector<float> &va)
{
  int nv = varray.size(0);
  int vs0 = varray.stride(0), vs1 = varray.stride(1);
  float *vp = varray.values();
  for (int k = 0 ; k < nv ; ++k)
    {
      va.push_back(vp[vs0*k]);
      va.push_back(vp[vs0*k+vs1]);
      va.push_back(vp[vs0*k+2*vs1]);
    }
}

// ----------------------------------------------------------------------------
//
static void subdivide_triangle(int i1, int i2, int i3, float elength,
			       Geometry &g)
{
  Indices e1, e2, e3;
  subdivide_edge(i2, i3, elength, g, e1);
  subdivide_edge(i3, i1, elength, g, e2);
  subdivide_edge(i1, i2, elength, g, e3);
  int n1 = e1.size(), n2 = e2.size(), n3 = e3.size();
  if (n1 == 2 && n2 == 2 && n3 == 2)
    {
      g.ta.push_back(i1);
      g.ta.push_back(i2);
      g.ta.push_back(i3);
    }
  else
    {
      int nmin = (n1 < n2 ? n1 : n2);
      nmin = (n3 < nmin ? n3 : nmin);
      if (n1 == nmin)
        triangle_strips(i1,i2,i3, e2,e3, elength, g);
      else if (n2 == nmin)
	triangle_strips(i2,i3,i1, e3,e1, elength, g);
      else
	triangle_strips(i3,i1,i2, e1,e2, elength, g);
    }
}

// ----------------------------------------------------------------------------
//
static void subdivide_edge(int i0, int i1, float elength, Geometry &g,
			   Indices &ev)
{
  Edge e;
  e.set_endpoints(i0, i1);
  std::set<Edge>::iterator ei = g.eset.find(e);
  if (ei != g.eset.end())
    (*ei).edge_vertices(ev, i0 > i1);
  else
    {
      int j0 = e.vindex1, j1 = e.vindex2;
      float d = distance(&g.va[3*j0], &g.va[3*j1]);
      int s = (int)(d / elength);
      float vf[3], nf[3];
      for (int i = 1 ; i <= s ; ++i)
	{
	  float f = float(i) / (s+1);
	  float *v0 = &g.va[3*j0], *v1 = &g.va[3*j1];
	  interpolated_vertex(v0, v1, f, &(vf[0]));
	  g.va.push_back(vf[0]);
	  g.va.push_back(vf[1]);
	  g.va.push_back(vf[2]);
	  float *n0 = &g.na[3*j0], *n1 = &g.na[3*j1];
	  interpolated_vertex(n0, n1, f, &(nf[0]));
	  g.na.push_back(nf[0]);
	  g.na.push_back(nf[1]);
	  g.na.push_back(nf[2]);
	}
      int n = g.va.size()/3;
      e.ev1 = n-s;
      e.ev2 = n-1;
      g.eset.insert(e);
      e.edge_vertices(ev, i0 > i1);
    }
}

// ----------------------------------------------------------------------------
//
static float distance(float *v0, float *v1)
{
  float dx = v0[0]-v1[0], dy = v0[1]-v1[1], dz = v0[2]-v1[2];
  return sqrt(dx*dx + dy*dy + dz*dz);
}

// ----------------------------------------------------------------------------
//
static void triangle_strips(int i1, int i2, int i3, Indices &e2, Indices &e3,
			    float elength, Geometry &g)
{
  int n2 = e2.size(), n3 = e3.size();
  int l2 = n2-1;
  g.ta.push_back(i1);
  g.ta.push_back(e3[1]);
  g.ta.push_back(e2[l2-1]);
  int ns = (n2 < n3 ? n2 : n3) - 2;
  Indices ev1, ev2;
  for (int i  = 0 ; i < ns ; ++i)
    {
      subdivide_edge(e3[i+1], e2[l2-(i+1)], elength, g, ev1);
      subdivide_edge(e3[i+2], e2[l2-(i+2)], elength, g, ev2);
      stitch_strip(ev1, ev2, g);
    }
  if (n2 > n3)
    subdivide_triangle(e2[l2-(ns+1)], i2, i3, elength, g);
  else if (n3 > n2)
    subdivide_triangle(i3, e3[ns+1], i2, elength, g);
}

// ----------------------------------------------------------------------------
//
static void stitch_strip(Indices &e1, Indices &e2, Geometry &g)
{
  int n1 = e1.size(), n2 = e2.size();
  int k1 = 0, k2 = 0;
  while (k1+1 < n1 || k2+1 < n2)
    {
      int i1 = e1[k1], i2 = e2[k2], i3, t = 0;
      if (k1+1 < n1 && k2+1 < n2)
	{
	  float d1 = distance(&g.va[3*i2], &g.va[3*e1[k1+1]]);
	  float d2 = distance(&g.va[3*i1], &g.va[3*e2[k2+1]]);
	  t = (d2 < d1 ? 2 : 1);
	}
      else 
	t = (k1+1 < n1 ? 1 : 2);
      if (t == 1)
	{
	  i3 = e1[k1+1];
	  k1 += 1;
	}
      else
	{
	  i3 = e2[k2+1];
	  k2 += 1;
	}
      g.ta.push_back(i1);
      g.ta.push_back(i2);
      g.ta.push_back(i3);
    }
}
