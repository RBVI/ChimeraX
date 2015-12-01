// ----------------------------------------------------------------------------
// Compute the portion of a plane inside a given surface.
//
//#include <iostream>			// use std::cerr for debugging

#include <Python.h>			// Use PyObject

#include "border.h"			// Use Vertices, Loops
#include "pythonarray.h"		// use parse_float_n3_array(), ...
#include "rcarray.h"			// use FArray, IArray
#include "triangulate.h"		// use triangulate_polygon()

namespace Cap_Calculation
{

// ----------------------------------------------------------------------------
//
void compute_cap(float plane_normal[3], float plane_offset,
		 const FArray &varray, const IArray &tarray,  /* Surface */
		 Vertices &cap_vertex_positions,
		 Triangles &cap_triangle_vertex_indices)
{
  // Calculate plane intersection with surface.
  Loops loops;
  calculate_border(plane_normal, plane_offset, varray, tarray,
		   cap_vertex_positions, loops);

  // Triangulate polygonal region bounded by loops.
  triangulate_polygon(loops, plane_normal,
		      cap_vertex_positions, cap_triangle_vertex_indices);
}

}	// end of namespace Cap_Calculation

// ----------------------------------------------------------------------------
//
extern "C" PyObject *compute_cap(PyObject *, PyObject *args, PyObject *keywds)
{
  float normal[3], c;
  FArray varray;
  IArray tarray;
  const char *kwlist[] = {"normal", "offset", "vertices", "triangles", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&fO&O&"), (char **)kwlist,
				   parse_float_3_array, normal, &c,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray))
    return NULL;

  std::vector<float> cap_vertex_xyz;
  std::vector<int> cap_tv_indices;
  Cap_Calculation::compute_cap(normal, c, varray, tarray,
			       cap_vertex_xyz, cap_tv_indices);

  float *vxyz = (cap_vertex_xyz.size() == 0 ? NULL : &cap_vertex_xyz.front());
  int *ti = (cap_tv_indices.size() == 0 ? NULL : &cap_tv_indices.front());
  PyObject *py_cap_vertex_positions, *py_cap_triangle_vertex_indices;
  py_cap_vertex_positions = c_array_to_python(vxyz, cap_vertex_xyz.size()/3, 3);
  py_cap_triangle_vertex_indices = c_array_to_python(ti, cap_tv_indices.size()/3, 3);

  PyObject *geom = PyTuple_New(2);
  PyTuple_SetItem(geom, 0, py_cap_vertex_positions);
  PyTuple_SetItem(geom, 1, py_cap_triangle_vertex_indices);

  return geom;
}
