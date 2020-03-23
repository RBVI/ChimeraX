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

// -----------------------------------------------------------------------------
// Compute natural cubic spline through points in 3 dimensions.
//
#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use FArray, IArray
#include "spline.h"

static void solve_tridiagonal(double *y, int64_t n, double *temp);

// -----------------------------------------------------------------------------
// Match first and second derivatives at interval end-points and make second
// derivatives zero at two ends of path.
//
static void natural_cubic_spline(float *path, int64_t n, int segment_subdivisions,
				 float *spath, float *tangents)
{
  if (n == 0)
    return;
  if (n == 1)
    {
      spath[0] = path[0]; spath[1] = path[1]; spath[2] = path[2];
      tangents[0] = tangents[1] = tangents[2] = 0;
      return;
    }

  // Solve tridiagonal system to calculate spline
  double *b = new double [n];
  double *temp = new double [n];
  for (int a = 0 ; a < 3 ; ++a)
    {
      b[0] = 0;
      b[n-1] = 0;
      for (int64_t i = 1 ; i < n-1 ; ++i)
	b[i] = path[3*(i+1)+a] -2*path[3*i+a] + path[3*(i-1)+a];
      solve_tridiagonal(b,n,temp);
      int64_t k = 0;
      int div = segment_subdivisions;
      for (int64_t i = 0 ; i < n-1 ; ++i)
	{
	  int pc = (i < n-2 ? div + 1 : div + 2);
	  for (int s = 0 ; s < pc ; ++s)
	    {
	      double t = s / (div + 1.0);
	      double ct = path[3*(i+1)+a] - b[i+1];
	      double c1t = path[3*i+a] - b[i];
	      double u = 1-t;
	      spath[k+a] = b[i+1]*t*t*t + b[i]*u*u*u + ct*t + c1t*u;
	      tangents[k+a] = 3*b[i+1]*t*t - 3*b[i]*u*u + ct - c1t;
	      k += 3;
	    }
	}
    }
  delete [] b;
  delete [] temp;

  // normalize tangent vectors.
  int64_t ns = n + (n-1)*segment_subdivisions;
  int64_t ns3 = 3*ns;
  for (int64_t i = 0 ; i < ns3 ; i += 3)
    {
      float tx = tangents[i], ty = tangents[i+1], tz = tangents[i+2];
      float tn = sqrt(tx*tx + ty*ty + tz*tz);
      if (tn > 0)
	{
	  tangents[i] = tx/tn;
	  tangents[i+1] = ty/tn;
	  tangents[i+2] = tz/tn;
	}
    }
}

// -----------------------------------------------------------------------------
// Ax = y, y is modified and equals x on return.
// A is tridiagonal with ones on subdiagonal except 0 on last row
// ones on superdiagonal except 0 on last row
// and diagonal is 4 except for first and last row which are 1.
//
static void solve_tridiagonal(double *y, int64_t n, double *temp)
{
  temp[0] = 0.0;
  for (int64_t i = 1 ; i < n-1 ; ++i)
    {
      temp[i] = 1.0 / (4.0 - temp[i-1]);
      y[i] = (y[i] - y[i-1]) * temp[i];
    }
  for (int64_t i = n-2 ; i >= 0 ; --i)
    y[i] -= temp[i] * y[i+1];
}

// -----------------------------------------------------------------------------
//
const char *natural_cubic_spline_doc =
  "natural_cubic_spline(path, segment_subdivisions) -> spath, tangents\n"
  "\n"
  "Supported API\n"
  "Compute a natural cubic spline through path points in 3 dimensions,\n"
  "producing a finer set of points and tangent vectors at those points.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "path : n by 3 float array\n"
  "  points that spline will pass through.\n"
  "segment_subdivisions : int\n"
  "  place this number of additional points between every two consecutive\n"
  "  path points.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "spath : m by 3 float array\n"
  "  points on cubic spline including original points and subdivision points.\n"
  "tangents : m by 3 float array\n"
  "  tangent vector at point of returned path.\n";

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *natural_cubic_spline(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray path;
  int segment_subdivisions;
  const char *kwlist[] = {"path", "segment_subdivisions", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&i"),
				   (char **)kwlist,
				   parse_float_n3_array, &path,
				   &segment_subdivisions))
    return NULL;

  int64_t n = path.size(0);
  float *p = path.values();
  float *spath, *tangents;
  int64_t ns = (n > 1 ? n + (n-1)*segment_subdivisions : n);
  PyObject *spath_py = python_float_array(ns, 3, &spath);
  PyObject *tangents_py = python_float_array(ns, 3, &tangents);

  natural_cubic_spline(p, n, segment_subdivisions, spath, tangents);

  PyObject *pt = python_tuple(spath_py, tangents_py);
  return pt;
}

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
