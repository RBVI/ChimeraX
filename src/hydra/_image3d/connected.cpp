// ----------------------------------------------------------------------------
//
//#include <iostream>		// use std::cerr for debugging
#include <algorithm>		// use std::sort
#include <map>			// use std::map
#include <set>			// use std::set
#include <utility>		// use std::pair
#include <vector>		// use std::vector

#include <Python.h>		// use PyObject

#include "pythonarray.h"	// use array_from_python()
#include "rcarray.h"		// use IArray

static void connected_triangles(const IArray &tarray, int tindex,
				std::vector<int> &tlist);
static void triangle_vertices(int *tarray, int *tlist, int nt,
			      std::vector<int> &vlist);
static int maximum_triangle_vertex_index(const IArray &tarray);
static int calculate_components(const IArray &tarray, int *vmap, int vc);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *connected_triangles(PyObject *, PyObject *args, PyObject *keywds)
{
  IArray tarray;
  int tindex;
  const char *kwlist[] = {"triangles", "tindex", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&i"), (char **)kwlist,
				   parse_int_n3_array, &tarray,
				   &tindex))
    return NULL;

  std::vector<int> tlist;
  connected_triangles(tarray, tindex, tlist);
  int *ti = (tlist.size() == 0 ? NULL : &tlist.front());
  PyObject *py_tlist = c_array_to_python(ti, tlist.size());
  return py_tlist;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *triangle_vertices(PyObject *, PyObject *args, PyObject *keywds)
{
  IArray tarray, tlist;
  const char *kwlist[] = {"triangles", "tindices", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&"), (char **)kwlist,
				   parse_int_n3_array, &tarray,
				   parse_int_n_array, &tlist))
    return NULL;

  IArray ta = tarray.contiguous_array();
  IArray tl = tlist.contiguous_array();
  std::vector<int> vlist;
  triangle_vertices(ta.values(), tl.values(), tl.size(), vlist);
  int *vi = (vlist.size() == 0 ? NULL : &vlist.front());
  PyObject *py_vlist = c_array_to_python(vi, vlist.size());
  return py_vlist;
}

// ----------------------------------------------------------------------------
//
static void connected_triangles(const IArray &tarray, int tindex,
				std::vector<int> &tlist)
{
  // Create map of vertices to lowest number connected vertex.
  int vc = maximum_triangle_vertex_index(tarray) + 1;
  int *vmap = new int[vc];
  calculate_components(tarray, vmap, vc);
  int *ta = tarray.values(), ts = tarray.stride(0);
  int v = vmap[ta[ts*tindex]];

  // Find all triangles connected to tindex triangle.
  int nt = tarray.size(0);
  for (int t = 0 ; t < nt ; ++t)
    if (vmap[ta[ts*t]] == v)
      tlist.push_back(t);

  delete [] vmap;
}

// ----------------------------------------------------------------------------
//
static void triangle_vertices(int *tarray, int *tlist, int nt,
			      std::vector<int> &vlist)
{
  std::set<int> vset;
  for (int k = 0 ; k < nt ; ++k)
    {
      int t = tlist[k];
      vset.insert(tarray[3*t]);
      vset.insert(tarray[3*t+1]);
      vset.insert(tarray[3*t+2]);
    }
  vlist.insert(vlist.begin(), vset.begin(), vset.end());
  std::sort(vlist.begin(), vlist.end());
}

// ----------------------------------------------------------------------------
//
class Surface_Pieces
{
public:
  Surface_Pieces(const IArray &tarray);
  ~Surface_Pieces();
  typedef std::pair<std::vector<int>*, std::vector<int>*> Surface_Piece;
  std::vector<Surface_Piece> pieces;
};

// ----------------------------------------------------------------------------
//
Surface_Pieces::Surface_Pieces(const IArray &tarray)
{
  if (tarray.size() == 0)
    return;

  // Create map of vertices to lowest number connected vertex.
  int vc = maximum_triangle_vertex_index(tarray) + 1;
  int *vmap = new int[vc];
  int cc = calculate_components(tarray, vmap, vc);

  // Allocate surface piece vertex and triangle index arrays.
  for (int c = 0 ; c < cc ; ++c)
    pieces.push_back(Surface_Piece(new std::vector<int>, new std::vector<int>));

  // Fill vertex index piece arrays.
  for (int v = 0 ; v < vc ; ++v)
    if (vmap[v] < vc)
      pieces[vmap[v]].first->push_back(v);

  // Fill triangle index piece arrays.
  int tc = tarray.size(0), s0 = tarray.stride(0);
  const int *tv = tarray.values();
  for (int t = 0 ; t < tc ; ++t)
    pieces[vmap[tv[s0*t]]].second->push_back(t);

  delete [] vmap;
}

// ----------------------------------------------------------------------------
//
Surface_Pieces::~Surface_Pieces()
{
  int pc = static_cast<int>(pieces.size());
  for (int p = 0 ; p < pc ; ++p)
    {
      delete pieces[p].first;
      delete pieces[p].second;
    }
}

// ----------------------------------------------------------------------------
//
static int maximum_triangle_vertex_index(const IArray &tarray)
{
  int vmax = 0;
  int tc = tarray.size(0), s0 = tarray.stride(0), s1 = tarray.stride(1);
  const int *tv = tarray.values();
  for (int t = 0 ; t < tc ; ++t)
    {
      int v0 = tv[s0*t], v1 = tv[s0*t+s1], v2 = tv[s0*t+2*s1];
      if (v0 > vmax) vmax = v0;
      if (v1 > vmax) vmax = v1;
      if (v2 > vmax) vmax = v2;
    }
  return vmax;
}

// ----------------------------------------------------------------------------
//
inline int min_connected_vertex(int v, int *vmap)
{
  int v0 = v;
  while (vmap[v0] < v0)
    v0 = vmap[v0];
  // Collapse chain for efficiency of future traversals.
  for (int v1 = v ; v1 > v0 ; v1 = vmap[v1])
    vmap[v1] = v0;
  return v0;
}

// ----------------------------------------------------------------------------
//
static int calculate_components(const IArray &tarray, int *vmap, int vc)
{
  for (int v = 0 ; v < vc ; ++v)
    vmap[v] = vc;

  int tc = tarray.size(0);
  int s0 = tarray.stride(0), s1 = tarray.stride(1);
  const int *tv = tarray.values();
  for (int t = 0 ; t < tc ; ++t)
    {
      int v0 = min_connected_vertex(tv[s0*t], vmap);
      int v1 = min_connected_vertex(tv[s0*t+s1], vmap);
      int v2 = min_connected_vertex(tv[s0*t+2*s1], vmap);
      int v01 = (v0 < v1 ? v0 : v1);
      int vmin = (v2 < v01 ? v2 : v01);
      vmap[v0] = vmap[v1] = vmap[v2] = vmin;
    }

  // Make each vertex map to a connected component number 0,1,2,....
  int cc = 0;
  for (int v = 0 ; v < vc ; ++v)
    {
      int vm = vmap[v];
      if (vm < v)
	vmap[v] = vmap[vm];
      else if (vm == v)
	vmap[v] = cc++;
    }

  return cc;
}

// ----------------------------------------------------------------------------
//
static PyObject *python_array(std::vector<int> &v)
{
  return c_array_to_python(&v[0], static_cast<int>(v.size()));
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *connected_pieces(PyObject *, PyObject *args, PyObject *keywds)
{
  IArray tarray;
  const char *kwlist[] = {"triangles", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&"), (char **)kwlist,
				   parse_int_n3_array, &tarray))
    return NULL;

  Surface_Pieces sp(tarray);
  int pc = static_cast<int>(sp.pieces.size());
  PyObject *plist = PyTuple_New(pc);
  for (int c = 0 ; c < pc ; ++c)
    {
      PyObject *vt = PyTuple_New(2);
      PyTuple_SetItem(vt, 0, python_array(*sp.pieces[c].first));
      PyTuple_SetItem(vt, 1, python_array(*sp.pieces[c].second));
      PyTuple_SetItem(plist, c, vt);
    }
  return plist;
}
