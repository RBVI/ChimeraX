// Compute bounds of set of points.
#include <Python.h>		// use PyObject *

#include "parsepdb.h"		// use Atom
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

static bool atom_bounds(Atom *a, int n, float xyz_min[3], float xyz_max[3])
{
  float xmin, ymin, zmin, xmax, ymax, zmax;
  int c = 0;
  for (int i = 0 ; i < n ; ++i, ++a)
    if (a->atom_shown || a->ribbon_shown)
      {
	float r = a->radius;
	if (c == 0)
	  { xmin = a->x - r; ymin = a->y - r; zmin = a->z - r; xmax = a->x + r; ymax = a->y + r; zmax = a->z + r; }
	else
	  {
	    float x = a->x - r, y = a->y - r, z = a->z -r;
	    if (x < xmin) xmin = x;
	    if (y < ymin) ymin = y;
	    if (z < zmin) zmin = z;
	    x = a->x + r; y = a->y + r; z = a->z +r;
	    if (x > xmax) xmax = x;
	    if (y > ymax) ymax = y;
	    if (z > zmax) zmax = z;
	  }
	c += 1;
      }
  if (c == 0)
    return false;

  xyz_min[0] = xmin; xyz_min[1] = ymin; xyz_min[2] = zmin; 
  xyz_max[0] = xmax; xyz_max[1] = ymax; xyz_max[2] = zmax;

  return true;
}

extern "C" PyObject *atom_bounds(PyObject *s, PyObject *args, PyObject *keywds)
{
  CArray atoms;
  const char *kwlist[] = {"atoms", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_string_array, &atoms))
    return NULL;

  if (atoms.dimension() != 2)
    {
      PyErr_SetString(PyExc_TypeError, "atom_bounds(): array must be 2 dimensional");
      return NULL;
    }
  if (atoms.size(1) != sizeof(Atom))
    {
      PyErr_SetString(PyExc_TypeError, "atom_bounds(): Wrong atom object size");
      return NULL;
    }
  if (!atoms.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "atom_bounds(): array must be contiguous");
      return NULL;
    }

  Atom *aa = (Atom *)atoms.values();
  int n = atoms.size(0);

  float xyz_min[3], xyz_max[3];
  if (!atom_bounds(aa, n, xyz_min, xyz_max))
    return python_none();

  return python_tuple(c_array_to_python(&xyz_min[0], 3),
		      c_array_to_python(&xyz_max[0], 3));
}
