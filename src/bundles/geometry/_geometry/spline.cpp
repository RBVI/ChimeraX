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
static void natural_cubic_spline(float *path, int64_t n, int64_t dim,
				 int fixed_segment_subdivisions, int *segment_subdivisions,
				 float *spath, float *tangents)
{
  if (n == 0)
    return;
  if (n == 1)
    {
      for (int64_t a = 0 ; a < dim ; ++a)
	{
	  spath[a] = path[a];
	  if (tangents)
	    tangents[a] = 0;
	}
      return;
    }

  // Solve tridiagonal system to calculate spline
  double *b = new double [n];
  double *temp = new double [n];
  for (int64_t a = 0 ; a < dim ; ++a)
    {
      b[0] = 0;
      b[n-1] = 0;
      for (int64_t i = 1 ; i < n-1 ; ++i)
	b[i] = path[dim*(i+1)+a] -2*path[dim*i+a] + path[dim*(i-1)+a];
      solve_tridiagonal(b,n,temp);
      int64_t k = 0;
      for (int64_t i = 0 ; i < n-1 ; ++i)
	{
	  int div = (segment_subdivisions ?
		     segment_subdivisions[i] : fixed_segment_subdivisions);
	  int pc = (i < n-2 ? div + 1 : div + 2);
	  for (int s = 0 ; s < pc ; ++s)
	    {
	      double t = s / (div + 1.0);
	      double ct = path[dim*(i+1)+a] - b[i+1];
	      double c1t = path[dim*i+a] - b[i];
	      double u = 1-t;
	      spath[k+a] = b[i+1]*t*t*t + b[i]*u*u*u + ct*t + c1t*u;
	      if (tangents)
		tangents[k+a] = 3*b[i+1]*t*t - 3*b[i]*u*u + ct - c1t;
	      k += dim;
	    }
	}
    }
  delete [] b;
  delete [] temp;

  // normalize tangent vectors.
  if (tangents)
    {
      int64_t ns = n;
      if (segment_subdivisions)
	for (int64_t i = 0 ; i < n-1 ; ++i)
	  ns += segment_subdivisions[i];
      else
	ns += (n-1)*fixed_segment_subdivisions;
      int64_t nsd = dim*ns;
      for (int64_t i = 0 ; i < nsd ; i += dim)
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
}

// -----------------------------------------------------------------------------
// Ax = y, y is modified and equals x on return.
// A is tridiagonal with ones on subdiagonal except 0 on last row
// ones on superdiagonal except 0 on first row
// and diagonal is 4 except for first and last row which are 1.
//
// | 1 0 0          |
// | 1 4 1 0        |
// | 0 1 4 1 0      |
// |  0 1 4 1 0     |
// |        .       | x = y
// |      0 1 4 1 0 |
// |        0 1 4 1 |
// |            0 1 |
//
static void solve_tridiagonal(double *y, int64_t n, double *temp)
{
  // First eliminate subdiagonal and make dialog all 1.
  // temp[i] becomes the superdiagonal, and y[i] the new y.
  temp[0] = 0.0;
  for (int64_t i = 1 ; i < n-1 ; ++i)
    {
      temp[i] = 1.0 / (4.0 - temp[i-1]);
      y[i] = (y[i] - y[i-1]) * temp[i];
    }
  // Then eliminate the superdialogonal, leaving just diagonal all 1.
  // Now y is the sought solution x.
  for (int64_t i = n-2 ; i >= 0 ; --i)
    y[i] -= temp[i] * y[i+1];
}

// -----------------------------------------------------------------------------
//
static PyObject *natural_cubic_spline_subdivisions_array(PyObject *args, PyObject *keywds)
{
  FArray path;
  IArray segment_subdiv;
  int return_tangents = 1;
  const char *kwlist[] = {"path", "segment_subdivisions", "tangents", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&|p"),
				   (char **)kwlist,
				   parse_float_2d_array, &path,
				   parse_int_n_array, &segment_subdiv,
				   &return_tangents))
    return NULL;

  // Check that subdivisions array has correct length.
  if (path.size(0) > 0 && segment_subdiv.size(0)+1 != path.size(0))
    {
      PyErr_Format(PyExc_ValueError,
		   "natural_cubic_spline(): segment subdivision array (%s)must have length one less than number of interpolated points (%s)",
		   segment_subdiv.size_string(0).c_str(), path.size_string(0).c_str());
      return NULL;
    }

					
  IArray sdiv = segment_subdiv.contiguous_array();
  int *segment_subdivisions = sdiv.values();
  int fixed_segment_subdivisions = 0;	// Not used.

  int64_t n = path.size(0), np = path.size(1);
  FArray cpath = path.contiguous_array();
  float *p = cpath.values();
  float *spath, *tangents = NULL;
  int64_t ns = n;
  for (int64_t i = 0 ; i < n-1 ; ++i)
    ns += segment_subdivisions[i];
  PyObject *result;
  PyObject *spath_py = python_float_array(ns, np, &spath);
  if (return_tangents)
    {
      PyObject *tangents_py = python_float_array(ns, np, &tangents);
      result = python_tuple(spath_py, tangents_py);
    }
  else
    result = spath_py;

  natural_cubic_spline(p, n, np, fixed_segment_subdivisions,
		       segment_subdivisions, spath, tangents);
      
  return result;
}

// -----------------------------------------------------------------------------
//
const char *natural_cubic_spline_doc =
  "natural_cubic_spline(path, segment_subdivisions) -> spath, tangents\n"
  "\n"
  "Supported API\n"
  "Compute a natural cubic spline through path points in M dimensions,\n"
  "producing a finer set of points and tangent vectors at those points.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "path : N by M float array\n"
  "  N points in M-dimensions that spline will pass through.\n"
  "segment_subdivisions : int or int array of length n-1\n"
  "  place this number of additional points between every two consecutive\n"
  "  path points.  If an array is given each pair of path points can have\n"
  "  a different number of subdivisions.\n"
  "tangents : bool\n"
  "  whether spline path tangents are returned, default true.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "spath : m by M float array\n"
  "  points on cubic spline including original points and subdivision points.\n"
  "tangents : m by M float array\n"
  "  tangent vector at point of returned path.\n";

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *natural_cubic_spline(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray path;
  int segment_subdivisions;
  int return_tangents = 1;
  const char *kwlist[] = {"path", "segment_subdivisions", "tangents", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&i|p"),
				   (char **)kwlist,
				   parse_float_2d_array, &path,
				   &segment_subdivisions,
				   &return_tangents))
    {
      PyErr_Clear();
      return natural_cubic_spline_subdivisions_array(args, keywds);
    }

  int64_t n = path.size(0), np = path.size(1);
  FArray cpath = path.contiguous_array();
  float *p = cpath.values();
  int *segment_subdivisions_array = NULL;
  float *spath, *tangents = NULL;
  int64_t ns = (n > 1 ? n + (n-1)*segment_subdivisions : n);
  PyObject *result;
  PyObject *spath_py = python_float_array(ns, np, &spath);
  if (return_tangents)
    {
      PyObject *tangents_py = python_float_array(ns, np, &tangents);
      result = python_tuple(spath_py, tangents_py);
    }
  else
    result = spath_py;

  natural_cubic_spline(p, n, np, segment_subdivisions,
		       segment_subdivisions_array, spath, tangents);

  return result;
}
