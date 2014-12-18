// -----------------------------------------------------------------------------
// Compute natural cubic spline through points in 3 dimensions.
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray, IArray

static void solve_tridiagonal(double *y, int n, double *temp);

// -----------------------------------------------------------------------------
// Match first and second derivatives at interval end-points and make second
// derivatives zero at two ends of path.
//
static void natural_cubic_spline(float *path, int n, int segment_subdivisions,
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
      for (int i = 1 ; i < n-1 ; ++i)
	b[i] = path[3*(i+1)+a] -2*path[3*i+a] + path[3*(i-1)+a];
      solve_tridiagonal(b,n,temp);
      int k = 0;
      int div = segment_subdivisions;
      for (int i = 0 ; i < n-1 ; ++i)
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
  int ns = n + (n-1)*segment_subdivisions;
  int ns3 = 3*ns;
  for (int i = 0 ; i < ns3 ; i += 3)
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
static void solve_tridiagonal(double *y, int n, double *temp)
{
  temp[0] = 0.0;
  for (int i = 1 ; i < n-1 ; ++i)
    {
      temp[i] = 1.0 / (4.0 - temp[i-1]);
      y[i] = (y[i] - y[i-1]) * temp[i];
    }
  for (int i = n-2 ; i >= 0 ; --i)
    y[i] -= temp[i] * y[i+1];
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *natural_cubic_spline(PyObject *s, PyObject *args, PyObject *keywds)
{
  FArray path;
  int segment_subdivisions;
  const char *kwlist[] = {"path", "segment_subdivisions", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&i"),
				   (char **)kwlist,
				   parse_float_n3_array, &path,
				   &segment_subdivisions))
    return NULL;

  int n = path.size(0);
  float *p = path.values();
  float *spath, *tangents;
  int ns = (n > 1 ? n + (n-1)*segment_subdivisions : n);
  PyObject *spath_py = python_float_array(ns, 3, &spath);
  PyObject *tangents_py = python_float_array(ns, 3, &tangents);

  natural_cubic_spline(p, n, segment_subdivisions, spath, tangents);

  PyObject *pt = python_tuple(spath_py, tangents_py);
  return pt;
}
