// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Transform points with shift, scale and linear operations.
//
#include <Python.h>			// use PyObject
#include <math.h>			// use sqrtf()

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use call_template_function()

// -----------------------------------------------------------------------------
//
static void cylinder_rotations(float *axyz0, float *axyz1, int n, float *radii,
			       float *rot44)
{
  for (int i = 0 ; i < n ; ++i)
    {
      float vx = *axyz1++ - *axyz0++;
      float vy = *axyz1++ - *axyz0++;
      float vz = *axyz1++ - *axyz0++;
      float d = sqrtf(vx*vx + vy*vy + vz*vz);
      if (d == 0)
	{ vx = vy = 0 ; vz = 1; }
      else
	{ vx /= d; vy /= d; vz /= d; }

      float c = vz, c1;
      if (c <= -1)
	c1 = 0;       // Degenerate -z axis case.
      else
	c1 = 1.0/(1+c);

      float wx = -vy, wy = vx;
      float cx = c1*wx, cy = c1*wy;
      float r = *radii++;
      float h = d;

      *rot44++ = r*(cx*wx + c);
      *rot44++ = r*cy*wx;
      *rot44++ = -r*wy;
      *rot44++ = 0;

      *rot44++ = r*cx*wy;
      *rot44++ = r*(cy*wy + c);
      *rot44++ = r*wx;
      *rot44++ = 0;

      *rot44++ = h*wy;
      *rot44++ = -h*wx;
      *rot44++ = h*c;
      *rot44++ = 0;

      *rot44++ = 0;
      *rot44++ = 0;
      *rot44++ = 0;
      *rot44++ = 1;
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *cylinder_rotations(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray xyz0, xyz1, radii, rot44;
  const char *kwlist[] = {"xyz0", "xyz1", "radii", "rot44", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &xyz0,
				   parse_float_n3_array, &xyz1,
				   parse_float_n_array, &radii,
				   parse_writable_float_3d_array, &rot44))
    return NULL;

  int n = xyz0.size(0);
  if (xyz1.size(0) != n || radii.size(0) != n)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end-point and radii arrays must have same size, got %d and %d",
			n, xyz1.size(0), radii.size(0));
  if (rot44.size(0) != n || rot44.size(1) != 4 || rot44.size(2) != 4)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder rotations wrong size, got %d %d %d, expected %d 4 4",
			rot44.size(0), rot44.size(1), rot44.size(2), n);
  if (!xyz0.is_contiguous() || !xyz1.is_contiguous() || !radii.is_contiguous()
      || !rot44.is_contiguous())
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end point, radii or rotation array not contiguous.");

  cylinder_rotations(xyz0.values(), xyz1.values(), n, radii.values(), rot44.values());

  return python_none();
}
