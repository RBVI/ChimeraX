// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

#include <vector>			// use std::vector
#include <algorithm>			// use std::unique, std::sort

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use Numeric_Array, Array<T>

#define TRIANGLE_DISPLAY_MASK 8
#define EDGE0_DISPLAY_MASK 1
#define ALL_EDGES_DISPLAY_MASK 7

namespace Map_Cpp
{

static_assert(sizeof(int) * 2 == sizeof(size_t),
              "Size of int is not half of size_t - masked_edges needs a re-think!");


size_t edge_hash (int i0, int i1) noexcept
{
  return ((size_t)i0 << 32 | i1); 
}

// ----------------------------------------------------------------------------
// Find edges of displayed triangles.  Edges that appear in 2 or more triangles
// are only listed once.
//
static IArray calculate_masked_edges(const IArray &triangles,
				     const BArray &tmask, const BArray &emask)
{
  std::vector< size_t > edges;

  unsigned char *show_t = (tmask.size() > 0 ? tmask.values() : NULL);
  unsigned char *show_e = (emask.size() > 0 ? emask.values() : NULL);
  size_t n = triangles.size(0);

  int *tarray = triangles.values();
  for (size_t k = 0 ; k < n ; ++k, tarray += 3)
    {
      if (show_t == NULL || show_t[k])
	{
	  char ebits = (show_e == NULL ? 7 : show_e[k]);
	  for (int j = 0 ; j < 3 ; ++j)
	    if (ebits & (EDGE0_DISPLAY_MASK << j))
	      {
		int i0 = tarray[j], i1 = tarray[(j+1)%3];
		edges.push_back(i0 < i1 ? edge_hash(i0, i1) : edge_hash(i1, i0));
	      }
	}
    }

  // Sort and unique are 3x faster than using std::set or std::unordered_set
  // to remove duplicates.  Details in ChimeraX ticket #6243.
  std::sort(edges.begin(), edges.end());
  auto last = std::unique(edges.begin(), edges.end());
  edges.erase(last, edges.end());

  int64_t size[2] = {(int64_t)edges.size(), 2};
  IArray masked_edges(2, size);
  int *eiarray = masked_edges.values();
  for (auto ei = edges.begin() ; ei != edges.end() ; ++ei)
    {
      int i0 = (*ei)>>32;
      int i1 = (*ei)&0x00000000FFFFFFFF;
      *eiarray = i0; eiarray += 1;
      *eiarray = i1; eiarray += 1;
    }

  return masked_edges;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
masked_edges(PyObject *, PyObject *args, PyObject *keywds)
{
  IArray triangles;
  BArray tmask, emask;
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
