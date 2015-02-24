// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Count points in grid cells.
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use parse_float_n3_array(), ...
#include "rcarray.h"			// use FArray

namespace Map_Cpp
{

// ----------------------------------------------------------------------------
//
static void fill_occupancy_map(const FArray &xyz_list,
			       float xyz_origin[3], float xyz_step[3],
			       FArray &grid)
{
  int n = xyz_list.size(0);
  float x0 = xyz_origin[0], y0 = xyz_origin[1], z0 = xyz_origin[2];
  float xs = xyz_step[0], ys = xyz_step[1], zs = xyz_step[2];
  int s0 = xyz_list.stride(0), s1 = xyz_list.stride(1);
  float *xyz = xyz_list.values();
  int sk = grid.stride(0), sj = grid.stride(1), si = grid.stride(2);
  int szk = grid.size(0), szj = grid.size(1), szi = grid.size(2);
  float *g = grid.values();
  for (int m = 0 ; m < n ; ++m)
    {
      float x = xyz[s0*m], y = xyz[s0*m+s1], z = xyz[s0*m+2*s1];
      float ri = (x-x0)/xs, rj = (y-y0)/ys, rk = (z-z0)/zs;
      int i = static_cast<int>(ri);
      int j = static_cast<int>(rj);
      int k = static_cast<int>(rk);
      if (i >= 0 && i+1 < szi &&
	  j >= 0 && j+1 < szj &&
	  k >= 0 && k+1 < szk)
	{
	  float fi = ri-i, fj = rj-j, fk = rk-k;
	  int base = k*sk + j*sj + i*si;
	  g[base] += (1-fk)*(1-fj)*(1-fi);
	  g[base + si] += (1-fk)*(1-fj)*fi;
	  g[base + sj] += (1-fk)*fj*(1-fi);
	  g[base + sj + si] += (1-fk)*fj*fi;
	  g[base + sk] += fk*(1-fj)*(1-fi);
	  g[base + sk + si] += fk*(1-fj)*fi;
	  g[base + sk + sj] += fk*fj*(1-fi);
	  g[base + sk + sj + si] += fk*fj*fi;
	}
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *fill_occupancy_map(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points, map;
  float origin[3], step[3];
  const char *kwlist[] = {"points", "origin", "step", "map", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&O&"), (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_3_array, &origin,
				   parse_float_3_array, &step,
				   parse_writable_float_3d_array, &map))
    return NULL;

  fill_occupancy_map(points, origin, step, map);

  Py_INCREF(Py_None);
  return Py_None;
}

}	// End namespace Map_Cpp
