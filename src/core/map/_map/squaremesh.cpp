// ----------------------------------------------------------------------------
// Return integer array having length equal to length of triangle array
// with bit values indicating which edges that are parallel to one of the
// principle planes (xy, yz and xz).
//
// The returned array is for use as an argument to Surface_Renderer
// set_triangle_and_edge_mask().
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray, IArray

// ----------------------------------------------------------------------------
//
static void principle_plane_edges(const FArray &varray, const IArray &tarray,
				  IArray &barray);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
principle_plane_edges(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray varray;
  IArray tarray, barray;
  const char *kwlist[] = {"vertices", "triangles", "mask", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray,
				   parse_writable_int_n_array, &barray))
    return NULL;

  if (barray.size(0) != tarray.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "principle_plane_edges(): triangle array and mask array have unequal sizes");
      return NULL;
    }

  principle_plane_edges(varray, tarray, barray);

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static void principle_plane_edges(const FArray &varray, const IArray &tarray,
				  IArray &barray)
{
  int show_triangle_bit = 8;
  int edge_01_mask = 1, edge_12_mask = 2, edge_20_mask = 4;

  int *ta = tarray.values(), ts0 = tarray.stride(0), ts1 = tarray.stride(1);
  float *va = varray.values();
  int vs0 = varray.stride(0), vs1 = varray.stride(1);
  int yo = vs1, zo = 2*vs1;
  int *ba = barray.values();
  int n = tarray.size(0);
  for (int t = 0 ; t < n ; ++t)
    {
      int t0 = ts0 * t;
      int i0 = ta[t0], i1 = ta[t0+ts1], i2 = ta[t0+2*ts1];
      float *v0 = &va[vs0*i0], *v1 = &va[vs0*i1], *v2 = &va[vs0*i2];
      int emask = show_triangle_bit;
      if (v0[0] == v1[0] || v0[yo] == v1[yo] || v0[zo] == v1[zo])
	emask |= edge_01_mask;
      if (v1[0] == v2[0] || v1[yo] == v2[yo] || v1[zo] == v2[zo])
	emask |= edge_12_mask;
      if (v2[0] == v0[0] || v2[yo] == v0[yo] || v2[zo] == v0[zo])
	emask |= edge_20_mask;
      ba[t] = emask;
    }
}
