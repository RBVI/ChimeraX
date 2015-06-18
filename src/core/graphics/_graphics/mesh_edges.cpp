// vi: set expandtab shiftwidth=4 softtabstop=4:
#include <set>				// use std::set

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use Numeric_Array, Array<T>

#define TRIANGLE_DISPLAY_MASK 8
#define EDGE0_DISPLAY_MASK 1
#define ALL_EDGES_DISPLAY_MASK 7

namespace Map_Cpp
{

// ----------------------------------------------------------------------------
// Find edges of displayed triangles.  Edges that appear in 2 or more triangles
// are only listed once.
//
static IArray calculate_masked_edges(const IArray &triangles,
				     const CArray &tmask, const CArray &emask)
{
  std::set< std::pair<int,int> > edges;

  char *show_t = (tmask.size() > 0 ? tmask.values() : NULL);
  char *show_e = (emask.size() > 0 ? emask.values() : NULL);
  int n = triangles.size(0);
  int *tarray = triangles.values();
  for (int k = 0 ; k < n ; ++k, tarray += 3)
    {
      if (show_t == NULL || show_t[k])
	{
	  char ebits = (show_e == NULL ? 7 : show_e[k]);
	  for (int j = 0 ; j < 3 ; ++j)
	    if (ebits & (EDGE0_DISPLAY_MASK << j))
	      {
		int i0 = tarray[j], i1 = tarray[(j+1)%3];
		edges.insert(i0 < i1 ? std::pair<int,int>(i0, i1) :
			     std::pair<int,int>(i1, i0));
	      }
	}
    }

  int size[2] = {(int)edges.size(), 2};
  IArray masked_edges(2, size);
  int *eiarray = masked_edges.values();
  for (std::set< std::pair<int,int> >::iterator ei = edges.begin() ; 
       ei != edges.end() ; ++ei)
    {
      *eiarray = (*ei).first; eiarray += 1;
      *eiarray = (*ei).second; eiarray += 1;
    }

  return masked_edges;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
masked_edges(PyObject *, PyObject *args, PyObject *keywds)
{
  IArray triangles;
  CArray tmask, emask;
  const char *kwlist[] = {"triangles", "triangle_mask", "edge_mask", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&|O&O&"),
				   (char **)kwlist,
				   parse_int_n3_array, &triangles,
				   parse_uint8_n_array, &tmask,
				   parse_uint8_n_array, &emask))
    return NULL;

  if (tmask.size() > 0 && tmask.size(0) != triangles.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "masked_edges(): triangle mask array size does not "
		      "equal triangle array size");
      return NULL;
    }
  if (emask.size() > 0 && emask.size(0) != triangles.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "masked_edges(): edge mask array size does not "
		      "equal triangle array size");
      return NULL;
    }

  IArray edges = calculate_masked_edges(triangles, tmask, emask);
  PyObject *edges_py = c_array_to_python(edges.values(), edges.size(0), edges.size(1));
  return edges_py;
}

} // namespace Map_Cpp
