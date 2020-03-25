// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use python_float_array

// -----------------------------------------------------------------------------
//
static void cubic_path(double *c, double tmin, double tmax, int n, float *coords, float *tangents)
{
  double step = (n > 1 ? (tmax - tmin) / (n-1) : 0);
  double x0 = c[0], x1 = c[1], x2 = c[2], x3 = c[3];
  double y0 = c[4], y1 = c[5], y2 = c[6], y3 = c[7];
  double z0 = c[8], z1 = c[9], z2 = c[10], z3 = c[11];
  for (int i = 0 ; i < n ; ++i)
    {
      double t = tmin + i*step;
      double t_2 = 2*t;
      double t2 = t*t;
      double t2_3 = 3*t2;
      double t3 = t*t2;
      *coords = x0 + t*x1 + t2*x2 + t3*x3; ++coords;
      *coords = y0 + t*y1 + t2*y2 + t3*y3; ++coords;
      *coords = z0 + t*z1 + t2*z2 + t3*z3; ++coords;
      float tx = x1 + t_2*x2 + t2_3*x3;
      float ty = y1 + t_2*y2 + t2_3*y3;
      float tz = z1 + t_2*z2 + t2_3*z3;
      float tn = sqrtf(tx*tx + ty*ty + tz*tz);
      if (tn != 0)
	{
	  tx /= tn; ty /= tn; tz /= tn;
	}
      *tangents = tx; ++tangents;
      *tangents = ty; ++tangents;
      *tangents = tz; ++tangents;
    }
}

// -----------------------------------------------------------------------------
//
const char *cubic_path_doc =
  "cubic_path(coeffs, tmin, tmax, num_points) -> coords, tangents\n"
  "\n"
  "Supported API\n"
  "Compute a path in 3D using x,y,z cubic polynomials.\n"
  "Polynomial coefficients are given in 3x4 matrix coeffs, 64-bit float.\n"
  "The path is computed from t = tmin to tmax with num_points points.\n"
  "Points on the path and normalized tangent vectors are returned.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "coeffs : 3 by 4 float64 array\n"
  "  x,y,z cubic polynomial coefficients c0 + c1*t + c2*t*t + c3*t*t*t.\n"
  "tmin : float64\n"
  "  minimum t value.\n"
  "tmax : float64\n"
  "  maximum t value.\n"
  "num_points : int\n"
  "  number of points.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "coords : n by 3 float array\n"
  "  points on cubic path.\n"
  "tangents : n by 3 float array\n"
  "  normalized tangent vectors at each point of path.\n";

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *cubic_path(PyObject *, PyObject *args, PyObject *keywds)
{
  double coeffs[12];
  double tmin, tmax;
  int num_points;
  const char *kwlist[] = {"coeffs", "tmin", "tmax", "num_points", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ddi"),
				   (char **)kwlist,
				   parse_double_3x4_array, &coeffs[0],
				   &tmin, &tmax, &num_points))
    return NULL;

  float *coords, *tangents;
  PyObject *coords_py = python_float_array(num_points, 3, &coords);
  PyObject *tangents_py = python_float_array(num_points, 3, &tangents);

  cubic_path(&coeffs[0], tmin, tmax, num_points, coords, tangents);

  PyObject *ct = python_tuple(coords_py, tangents_py);
  return ct;
}
