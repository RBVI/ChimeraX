// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject

#include <math.h>			// use sqrt()

#include <map>				// use std::map
#include <set>				// use std::set
#include <stdexcept>			// use std::runtime_error
#include <utility>			// use std::pair
#include <vector>			// use std::vector

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray, IArray

typedef std::pair<int,int> Edge;
typedef std::set<Edge> Edge_Set;
typedef std::vector<int> Vertex_Loop;
typedef std::vector<Vertex_Loop> Vertex_Loops;

static float enclosed_volume(FArray &varray, const IArray &tarray,
			     int *hole_count);
static float tetrahedron_signed_volume(float v0[3], float v1[3], float v2[3],
				       float v3[3]);
static bool is_surface_closed(const IArray &tarray);
static Edge_Set *boundary_edge_set(const IArray &tarray);
static Vertex_Loops *boundary_loops(const IArray &tarray);
static float cap_volume(float *v, const Vertex_Loops &vloops);
static void loop_center(float *v, const Vertex_Loop &vloop, float center[3]);
static float surface_area(FArray &varray, const IArray &tarray,
			  float *areas = NULL);
static float triangle_area(float v0[3], float v1[3], float v2[3]);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *enclosed_volume(PyObject *s, PyObject *args, PyObject *keywds)
{
  FArray varray;
  IArray tarray;
  const char *kwlist[] = {"vertex_array", "triangle_array", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&"), (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray))
    return NULL;

  int hole_count;
  float vol = enclosed_volume(varray, tarray, &hole_count);
  PyObject *vh = python_tuple(PyFloat_FromDouble(vol), PyLong_FromLong(hole_count));
  return vh;
}

// ----------------------------------------------------------------------------
//
static float enclosed_volume(FArray &varray, const IArray &tarray,
			     int *hole_count)
{
  Vertex_Loops *vloops = boundary_loops(tarray);
  if (vloops == NULL)
    { *hole_count = 0; return -1.0; }
  *hole_count = static_cast<int>(vloops->size());

  FArray vc = varray.contiguous_array();
  float *v = vc.values();

  IArray tc = tarray.contiguous_array();
  int m = tarray.size(0);
  int *tv = tc.values();

  float volume = 0;
  for (int t = 0 ; t < m ; ++t)
    {
      int i0 = tv[3*t], i1 = tv[3*t+1], i2 = tv[3*t+2];
      volume += tetrahedron_signed_volume(v, v+3*i0, v+3*i1, v+3*i2);
    }

  volume += cap_volume(v, *vloops);
  delete vloops;

  if (volume < 0)
    volume = -volume;    // Sign depends on triangle vertex order.

  return volume;
}

// ----------------------------------------------------------------------------
//
static float tetrahedron_signed_volume(float v0[3], float v1[3], float v2[3],
				       float v3[3])
{

  float e1x = v1[0]-v0[0], e1y = v1[1]-v0[1], e1z = v1[2]-v0[2];
  float e2x = v2[0]-v0[0], e2y = v2[1]-v0[1], e2z = v2[2]-v0[2];
  float e3x = v3[0]-v0[0], e3y = v3[1]-v0[1], e3z = v3[2]-v0[2];
  float v = (e1x * (e2y * e3z - e2z * e3y) +
	     e1y * (e2z * e3x - e2x * e3z) +
	     e1z * (e2x * e3y - e2y * e3x)) / 6.0;
  return v;
}

// ----------------------------------------------------------------------------
//
static bool is_surface_closed(const IArray &tarray)
{
  Edge_Set *eset = boundary_edge_set(tarray);
  bool closed = eset->empty();
  delete eset;
  return closed;
}

// ----------------------------------------------------------------------------
//
static Edge_Set *boundary_edge_set(const IArray &tarray)
{
  IArray tc = tarray.contiguous_array();
  int m = tarray.size(0);
  int *tv = tc.values();

  Edge_Set *eset = new Edge_Set;
  Edge_Set::iterator ei;
  for (int t = 0 ; t < m ; ++t)
    {
      int i0 = tv[3*t], i1 = tv[3*t+1], i2 = tv[3*t+2];
      if ((ei = eset->find(Edge(i1,i0))) != eset->end()) eset->erase(ei);
      else eset->insert(Edge(i0,i1));
      if ((ei = eset->find(Edge(i2,i1))) != eset->end()) eset->erase(ei);
      else eset->insert(Edge(i1,i2));
      if ((ei = eset->find(Edge(i0,i2))) != eset->end()) eset->erase(ei);
      else eset->insert(Edge(i2,i0));
    }
  return eset;
}

// ----------------------------------------------------------------------------
// Returns NULL if surface is not oriented.
//
static Vertex_Loops *boundary_loops(const IArray &tarray)
{
  Edge_Set *eset = boundary_edge_set(tarray);

  // Map one vertex to next along directed boundary.
  std::map<int,int> vmap;
  for (Edge_Set::iterator ei = eset->begin() ; ei != eset->end() ; ++ei)
    {
      if (vmap.find(ei->first) == vmap.end())
	vmap[ei->first] = ei->second;
      else
	{ delete eset; return NULL; }
    }
  delete eset;

  // Record boundary loops.
  Vertex_Loops *vloops = new std::vector<Vertex_Loop>;
  while (!vmap.empty())
    {
      std::map<int,int>::iterator vmi = vmap.begin();
      Vertex_Loop vloop;
      while (true)
	{
	  vloop.push_back(vmi->first);
	  int v = vmi->second;
	  vmap.erase(vmi);
	  if (v == vloop[0])
	    break;
	  vmi = vmap.find(v);
	  if (vmi == vmap.end())
	    { delete vloops; return NULL; }
	}
      vloops->push_back(vloop);
    }
  return vloops;
}

// ----------------------------------------------------------------------------
//
static float cap_volume(float *v, const Vertex_Loops &vloops)
{
  float volume = 0;
  for (Vertex_Loops::const_iterator li = vloops.begin() ; li != vloops.end() ; ++li)
    {
      const Vertex_Loop &vloop = *li;
      int n = static_cast<int>(vloop.size());
      if (n >= 3)
	{
	  float c[3];
	  loop_center(v, vloop, c);
	  for (int k = 0 ; k < n ; ++k)
	    {
	      int i0 = vloop[k], i1 = vloop[(k+1)%n];
	      volume += tetrahedron_signed_volume(v, c, v+3*i1, v+3*i0);
	    }
	}
    }
  return volume;
}

// ----------------------------------------------------------------------------
//
static void loop_center(float *v, const Vertex_Loop &vloop, float center[3])
{
  center[0] = center[1] = center[2] = 0;
  int n = static_cast<int>(vloop.size());
  for (int k = 0 ; k < n ; ++k)
    {
      int i = vloop[k];
      for (int a = 0 ; a < 3 ; ++a)
	center[a] += v[3*i+a];
    }
  center[0] /= n; center[1] /= n; center[2] /= n;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *surface_area(PyObject *s, PyObject *args, PyObject *keywds)
{
  FArray varray;
  IArray tarray;
  const char *kwlist[] = {"vertex_array", "triangle_array", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&"), (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray))
    return NULL;

  float area = surface_area(varray, tarray);
  PyObject *py_area = PyFloat_FromDouble(area);
  return py_area;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *vertex_areas(PyObject *s, PyObject *args, PyObject *keywds)
{
  FArray varray, areas;
  IArray tarray;
  const char *kwlist[] = {"vertex_array", "triangle_array", "areas", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&|O&"), (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray,
				   parse_writable_float_n_array, &areas))
    return NULL;

  bool make_areas = (areas.dimension() == 0);
  if (make_areas)
    {
      float *a;
      int n = varray.size(0);
      PyObject *pya = python_float_array(n, &a);
      for (int k = 0 ; k < n ; ++k)
	a[k] = 0;
      parse_writable_float_n_array(pya, &areas);
    }
  else
    {
      if (areas.size(0) != varray.size(0))
	{
	  PyErr_SetString(PyExc_TypeError,
			  "vertex_areas: return array size does not equal number of vertices");
	  return NULL;
	}
      if (!areas.is_contiguous())
	{
	  PyErr_SetString(PyExc_TypeError,
			  "vertex_areas: return array is not contiguous");
	  return NULL;
	}
    }

  surface_area(varray, tarray, areas.values());
  PyObject *py_areas = array_python_source(areas);
  return py_areas;
}

// ----------------------------------------------------------------------------
//
static float surface_area(FArray &varray, const IArray &tarray, float *areas)
{
  FArray vc = varray.contiguous_array();
  float *v = vc.values();

  IArray tc = tarray.contiguous_array();
  int m = tarray.size(0);
  int *tv = tc.values();

  float area = 0;
  if (areas)
    for (int t = 0 ; t < m ; ++t)
      {
	int i0 = tv[3*t], i1 = tv[3*t+1], i2 = tv[3*t+2];
	float a = triangle_area(v+3*i0, v+3*i1, v+3*i2);
	area += a;
	areas[i0] += a/3.0;
	areas[i1] += a/3.0;
	areas[i2] += a/3.0;
      }
  else
    for (int t = 0 ; t < m ; ++t)
      {
	int i0 = tv[3*t], i1 = tv[3*t+1], i2 = tv[3*t+2];
	area += triangle_area(v+3*i0, v+3*i1, v+3*i2);
      }

  return area;
}

// ----------------------------------------------------------------------------
//
static float triangle_area(float v0[3], float v1[3], float v2[3])
{
  float x1 = v1[0]-v0[0], y1 = v1[1]-v0[1], z1 = v1[2]-v0[2];
  float x2 = v2[0]-v0[0], y2 = v2[1]-v0[1], z2 = v2[2]-v0[2];
  float x12 = y1*z2-z1*y2, y12 = z1*x2-x1*z2, z12 = x1*y2-y1*x2;
  float a2 = x12*x12 + y12*y12 + z12*z12;
  float area = .5*sqrt(a2);
  return area;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *boundary_edges(PyObject *s, PyObject *args, PyObject *keywds)
{
  IArray tarray;
  const char *kwlist[] = {"triangle_array", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&"), (char **)kwlist,
				   parse_int_n3_array, &tarray))
    return NULL;

  Edge_Set *eset = boundary_edge_set(tarray);
  int *ea, e = 0;
  PyObject *edges = python_int_array(static_cast<int>(eset->size()), 2, &ea);
  for (Edge_Set::iterator ei = eset->begin() ; ei != eset->end() ; ++ei)
    {
      ea[e++] = ei->first;
      ea[e++] = ei->second;
    }
  delete eset;

  return edges;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *boundary_loops(PyObject *s, PyObject *args, PyObject *keywds)
{
  IArray tarray;
  const char *kwlist[] = {"triangle_array", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&"), (char **)kwlist,
				   parse_int_n3_array, &tarray))
    return NULL;

  Vertex_Loops *vloops = boundary_loops(tarray);
  PyObject *loopy = PyTuple_New(vloops->size());
  int l = 0;
  for (Vertex_Loops::iterator vi = vloops->begin() ;
       vi != vloops->end() ; ++vi, ++l)
    PyTuple_SetItem(loopy, l, c_array_to_python(*vi));
  delete vloops;

  return loopy;
}
