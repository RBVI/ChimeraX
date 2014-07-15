// Compute bounds of set of points.
#include <Python.h>		// use PyObject *

#include "pythonarray.h"	// use parse_double_n3_array, ...
#include "rcarray.h"		// use DArray

static void point_bounds(const FArray &points, float xyz_min[3], float xyz_max[3])
{
  int n = points.size(0);
  long s0 = points.stride(0), s1 = points.stride(1);
  float *pa = points.values();

  if (n > 0)
    for (int a = 0 ; a < 3 ; ++a)
      xyz_max[a] = xyz_min[a] = pa[a*s1];
  else
    for (int a = 0 ; a < 3 ; ++a)
      xyz_max[a] = xyz_min[a] = 0;

  for (int i = 0 ; i < n ; ++i)
    for (int a = 0 ; a < 3 ; ++a)
      {
	float x = pa[i*s0 + a*s1];
	if (x < xyz_min[a])
	  xyz_min[a] = x;
	else if (x > xyz_max[a])
	  xyz_max[a] = x;
      }
}

extern "C" PyObject *point_bounds(PyObject *s, PyObject *args, PyObject *keywds)
{
  FArray points;
  const char *kwlist[] = {"points", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&"), (char **)kwlist,
				   parse_float_n3_array, &points))
    return NULL;

  float xyz_min[3], xyz_max[3];
  point_bounds(points, xyz_min, xyz_max);

  return python_tuple(c_array_to_python(&xyz_min[0], 3),
		      c_array_to_python(&xyz_max[0], 3));
}
