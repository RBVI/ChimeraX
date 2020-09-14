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
