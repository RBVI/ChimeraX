// vi: set expandtab shiftwidth=4 softtabstop=4:
// -----------------------------------------------------------------------------
// Computations involving bounds
//
#include <Python.h>			// use PyObject
#include <vector>			// use std::vector

#include "pythonarray.h"		// use parse_float_n3_array, ...
#include "rcarray.h"			// use FArray

// -----------------------------------------------------------------------------
//
static void axes_sphere_bounds(const FArray &centers, const FArray &radii, const FArray &axes,
			       float *axes_bounds)
{
  long n = centers.size(0), na = axes.size(0);
  long cs0 = centers.stride(0), cs1 = centers.stride(1);
  long rs = radii.stride(0), as0 = axes.stride(0), as1 = axes.stride(1);
  float *ca = centers.values(), *ra = radii.values(), *aa = axes.values(), *ba = axes_bounds;
  for (long i = 0 ; i < n ; ++i)
    {
      float *ci = ca + i*cs0;
      float x = ci[0], y = ci[cs1], z = ci[2*cs1], r = ra[i*rs];
      for (long j = 0 ; j < na ; ++j)
	{
	  float *aj = aa + j*as0;
	  float d = x*aj[0] + y*aj[as1] + z*aj[2*as1];
	  float dm = d - r, dp = d + r;
	  if (dm < ba[2*j] || i == 0)
	    ba[2*j] = dm;
	  if (dp > ba[2*j+1] || i == 0)
	    ba[2*j+1] = dp;
	}
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *sphere_axes_bounds(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray centers, radii, axes;
  const char *kwlist[] = {"centers", "radii", "axes", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &centers,
				   parse_float_n_array, &radii,
				   parse_float_n3_array, &axes))
    return NULL;

  if (radii.size() != centers.size(0))
    return PyErr_Format(PyExc_ValueError, "Centers and radii arrays have different sizes %ld and %ld",
			centers.size(0), radii.size());

  float *axes_bounds;
  PyObject *bounds = python_float_array(axes.size(0), 2, &axes_bounds);
  axes_sphere_bounds(centers, radii, axes, axes_bounds);
  return bounds;
}

// -----------------------------------------------------------------------------
//
static void spheres_in_axes_bounds(const FArray &centers, const FArray &radii, const FArray &axes,
				   const FArray &axes_bounds, float padding, std::vector<int> &in)
{
  long n = centers.size(0), na = axes.size(0);
  long cs0 = centers.stride(0), cs1 = centers.stride(1);
  long rs = radii.stride(0), as0 = axes.stride(0), as1 = axes.stride(1);
  long bs0 = axes_bounds.stride(0), bs1 = axes_bounds.stride(1);
  float *ca = centers.values(), *ra = radii.values(), *aa = axes.values(), *ba = axes_bounds.values();
  for (long i = 0 ; i < n ; ++i)
    {
      float *ci = ca + i*cs0;
      float x = ci[0], y = ci[cs1], z = ci[2*cs1], r = ra[i*rs];
      bool jin = true;
      for (long j = 0 ; j < na ; ++j)
	{
	  float *aj = aa + j*as0;
	  float d = x*aj[0] + y*aj[as1] + z*aj[2*as1];
	  float bmin = ba[bs0*j], bmax = ba[bs0*j+bs1];
	  if (d+r < bmin-padding || d-r > bmax+padding)
	    {
	      jin = false;
	      break;
	    }
	}
      if (jin)
	in.push_back(static_cast<int>(i));
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *spheres_in_bounds(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray centers, radii, axes, axes_bounds;
  float padding;
  const char *kwlist[] = {"centers", "radii", "axes", "axes_bounds", "padding", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&f"),
				   (char **)kwlist,
				   parse_float_n3_array, &centers,
				   parse_float_n_array, &radii,
				   parse_float_n3_array, &axes,
				   parse_float_n2_array, &axes_bounds,
				   &padding))
    return NULL;

  if (radii.size() != centers.size(0))
    return PyErr_Format(PyExc_ValueError, "Centers and radii arrays have different sizes %ld and %ld",
			centers.size(0), radii.size());
  if (axes.size(0) != axes_bounds.size(0))
    return PyErr_Format(PyExc_ValueError, "Axes and axes bounds arrays have different sizes %ld and %ld",
			axes.size(0), axes_bounds.size(0));

  std::vector<int> in;
  spheres_in_axes_bounds(centers, radii, axes, axes_bounds, padding, in);
  PyObject *sib = c_array_to_python(in);
  return sib;
}

// -----------------------------------------------------------------------------
//
static bool axes_bounds_overlap(const FArray &bounds1, const FArray &bounds2, float padding)
{
  long n = bounds1.size(0);
  long b1s0 = bounds1.stride(0), b1s1 = bounds1.stride(1);
  long b2s0 = bounds2.stride(0), b2s1 = bounds2.stride(1);
  float *b1 = bounds1.values(), *b2 = bounds2.values();
  for (long i = 0 ; i < n ; ++i)
    {
      float b1min = b1[i*b1s0], b1max = b1[i*b1s0+b1s1];
      float b2min = b2[i*b2s0], b2max = b2[i*b2s0+b2s1];
      if (b1max + padding < b2min || b2max + padding < b1min)
	return false;
    }
  return true;
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *bounds_overlap(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray bounds1, bounds2;
  float padding;
  const char *kwlist[] = {"bounds1", "bounds2", "padding", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&f"),
				   (char **)kwlist,
				   parse_float_n2_array, &bounds1,
				   parse_float_n2_array, &bounds2,
				   &padding))
    return NULL;

  if (bounds1.size(0) != bounds2.size(0))
    return PyErr_Format(PyExc_ValueError, "Axes bounds arrays have different sizes %ld and %ld",
			bounds1.size(0), bounds2.size(0));

  bool overlap = axes_bounds_overlap(bounds1, bounds2, padding);
  return python_bool(overlap);
}

// -----------------------------------------------------------------------------
//
static void sphere_bounding_box(const FArray &centers, const FArray &radii, float *xyz_min, float *xyz_max)
{
  long n = centers.size(0);
  long cs0 = centers.stride(0), cs1 = centers.stride(1), rs = radii.stride(0);
  float *ca = centers.values(), *ra = radii.values();
  float xmin = 0, ymin = 0, zmin = 0, xmax = 0, ymax = 0, zmax = 0;
  for (long i = 0 ; i < n ; ++i, ca += cs0, ra += rs)
    {
      float x = ca[0], y = ca[cs1], z = ca[2*cs1], r = *ra;
      if (i == 0)
	{ xmin = x-r; xmax = x+r; ymin = y-r; ymax = y+r; zmin = z-r; zmax = z+r; }
      else
	{
	  if (x-r < xmin) xmin = x-r;
	  else if (x+r > xmax) xmax = x+r;
	  if (y-r < ymin) ymin = y-r;
	  else if (y+r > ymax) ymax = y+r;
	  if (z-r < zmin) zmin = z-r;
	  else if (z+r > zmax) zmax = z+r;
	}
    }
  xyz_min[0] = xmin; xyz_min[1] = ymin; xyz_min[2] = zmin;
  xyz_max[0] = xmax; xyz_max[1] = ymax; xyz_max[2] = zmax;
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *sphere_bounds(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray centers, radii;
  const char *kwlist[] = {"centers", "radii", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &centers,
				   parse_float_n_array, &radii))
    return NULL;

  if (radii.size() != centers.size(0))
    return PyErr_Format(PyExc_ValueError, "Centers and radii arrays have different sizes %ld and %ld",
			centers.size(0), radii.size());

  float *xyz_bounds;
  PyObject *bounds = python_float_array(2, 3, &xyz_bounds);
  sphere_bounding_box(centers, radii, xyz_bounds, xyz_bounds+3);
  return bounds;
}
