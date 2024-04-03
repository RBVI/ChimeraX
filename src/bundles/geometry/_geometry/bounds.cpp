// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// -----------------------------------------------------------------------------
// Computations involving bounds
//
#include <Python.h>			// use PyObject
#include <vector>			// use std::vector

#include <arrays/pythonarray.h>		// use parse_float_n3_array, ...
#include <arrays/rcarray.h>		// use FArray
#include "bounds.h"

// -----------------------------------------------------------------------------
//
static void axes_sphere_bounds(const FArray &centers, const FArray &radii, const FArray &axes,
			       float *axes_bounds)
{
  int64_t n = centers.size(0), na = axes.size(0);
  int64_t cs0 = centers.stride(0), cs1 = centers.stride(1);
  int64_t rs = radii.stride(0), as0 = axes.stride(0), as1 = axes.stride(1);
  float *ca = centers.values(), *ra = radii.values(), *aa = axes.values(), *ba = axes_bounds;
  for (int64_t i = 0 ; i < n ; ++i)
    {
      float *ci = ca + i*cs0;
      float x = ci[0], y = ci[cs1], z = ci[2*cs1], r = ra[i*rs];
      for (int64_t j = 0 ; j < na ; ++j)
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
    return PyErr_Format(PyExc_ValueError, "Centers and radii arrays have different sizes %s and %s",
			centers.size_string(0).c_str(), radii.size_string().c_str());

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
  int64_t n = centers.size(0), na = axes.size(0);
  int64_t cs0 = centers.stride(0), cs1 = centers.stride(1);
  int64_t rs = radii.stride(0), as0 = axes.stride(0), as1 = axes.stride(1);
  int64_t bs0 = axes_bounds.stride(0), bs1 = axes_bounds.stride(1);
  float *ca = centers.values(), *ra = radii.values(), *aa = axes.values(), *ba = axes_bounds.values();
  for (int64_t i = 0 ; i < n ; ++i)
    {
      float *ci = ca + i*cs0;
      float x = ci[0], y = ci[cs1], z = ci[2*cs1], r = ra[i*rs];
      bool jin = true;
      for (int64_t j = 0 ; j < na ; ++j)
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
    return PyErr_Format(PyExc_ValueError, "Centers and radii arrays have different sizes %s and %s",
			centers.size_string(0).c_str(), radii.size_string().c_str());
  if (axes.size(0) != axes_bounds.size(0))
    return PyErr_Format(PyExc_ValueError, "Axes and axes bounds arrays have different sizes %s and %s",
			axes.size_string(0).c_str(), axes_bounds.size_string(0).c_str());

  std::vector<int> in;
  spheres_in_axes_bounds(centers, radii, axes, axes_bounds, padding, in);
  PyObject *sib = c_array_to_python(in);
  return sib;
}

// -----------------------------------------------------------------------------
//
static bool axes_bounds_overlap(const FArray &bounds1, const FArray &bounds2, float padding)
{
  int64_t n = bounds1.size(0);
  int64_t b1s0 = bounds1.stride(0), b1s1 = bounds1.stride(1);
  int64_t b2s0 = bounds2.stride(0), b2s1 = bounds2.stride(1);
  float *b1 = bounds1.values(), *b2 = bounds2.values();
  for (int64_t i = 0 ; i < n ; ++i)
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
  int64_t n = centers.size(0);
  int64_t cs0 = centers.stride(0), cs1 = centers.stride(1), rs = radii.stride(0);
  float *ca = centers.values(), *ra = radii.values();
  float xmin = 0, ymin = 0, zmin = 0, xmax = 0, ymax = 0, zmax = 0;
  for (int64_t i = 0 ; i < n ; ++i, ca += cs0, ra += rs)
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
    return PyErr_Format(PyExc_ValueError, "Centers and radii arrays have different sizes %s and %s",
			centers.size_string(0).c_str(), radii.size_string().c_str());

  float *xyz_bounds;
  PyObject *bounds = python_float_array(2, 3, &xyz_bounds);
  sphere_bounding_box(centers, radii, xyz_bounds, xyz_bounds+3);
  return bounds;
}

// -----------------------------------------------------------------------------
//
static void points_within_planes(const FArray &points, const FArray &planes, unsigned char *pmask)
{
  float *p = points.values(), *pl = planes.values();
  int64_t n = points.size(0), np = planes.size(0), j;
  int64_t ps0 = points.stride(0), ps1 = points.stride(1);
  int64_t pls0 = planes.stride(0), pls1 = planes.stride(1);
  for (int64_t i = 0 ; i < n ; ++i)
    {
      for (j = 0 ; j < np ; ++j)
	if (p[i*ps0]*pl[j*pls0] + p[i*ps0+ps1]*pl[j*pls0+pls1] + p[i*ps0+2*pls1]*pl[j*pls0+2*pls1] + pl[j*pls0+3*pls1] < 0)
	    break;
      pmask[i] = (j < np ? 0 : 1);
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *points_within_planes(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points, planes;
  const char *kwlist[] = {"points", "planes", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_n4_array, &planes))
    return NULL;

  unsigned char *pmask;
  PyObject *pm = python_bool_array(points.size(0), &pmask);
  points_within_planes(points, planes, pmask);
  return pm;
}

// -----------------------------------------------------------------------------
//
static void points_bounding_box(const FArray &points, float *xyz_min, float *xyz_max)
{
  int64_t n = points.size(0);
  int64_t ps0 = points.stride(0), ps1 = points.stride(1);
  float *pa = points.values();
  float xmin = 0, ymin = 0, zmin = 0, xmax = 0, ymax = 0, zmax = 0;
  for (int64_t i = 0 ; i < n ; ++i, pa += ps0)
    {
      float x = pa[0], y = pa[ps1], z = pa[2*ps1];
      if (i == 0)
	{ xmin = xmax = x; ymin = ymax = y; zmin = zmax = z; }
      else
	{
	  if (x < xmin) xmin = x;
	  else if (x > xmax) xmax = x;
	  if (y < ymin) ymin = y;
	  else if (y > ymax) ymax = y;
	  if (z < zmin) zmin = z;
	  else if (z > zmax) zmax = z;
	}
    }
  xyz_min[0] = xmin; xyz_min[1] = ymin; xyz_min[2] = zmin;
  xyz_max[0] = xmax; xyz_max[1] = ymax; xyz_max[2] = zmax;
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *point_bounds(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points;
  const char *kwlist[] = {"points", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &points))
    return NULL;

  float *xyz_bounds;
  PyObject *bounds = python_float_array(2, 3, &xyz_bounds);
  points_bounding_box(points, xyz_bounds, xyz_bounds+3);
  return bounds;
}

// -----------------------------------------------------------------------------
//
static void point_copies_bounding_box(const FArray &points, const FArray &positions,
				      float *xyz_min, float *xyz_max)
{
  int64_t n = points.size(0), m = positions.size(0);
  int64_t ps0 = points.stride(0), ps1 = points.stride(1);
  int64_t pos0 = positions.stride(0), pos1 = positions.stride(1), pos2 = positions.stride(2);
  float *pa0 = points.values(), *poa = positions.values();
  float xmin = 0, ymin = 0, zmin = 0, xmax = 0, ymax = 0, zmax = 0;
  for (int64_t j = 0 ; j < m ; ++j, poa += pos0)
    {
      int64_t k0 = 0, k1 = pos2, k2 = 2*pos2, k3 = 3*pos2;
      float *poa0 = poa, *poa1 = poa + pos1, *poa2 = poa + 2*pos1;
      float p00 = poa0[k0], p01 = poa0[k1], p02 = poa0[k2], p03 = poa0[k3];
      float p10 = poa1[k0], p11 = poa1[k1], p12 = poa1[k2], p13 = poa1[k3];
      float p20 = poa2[k0], p21 = poa2[k1], p22 = poa2[k2], p23 = poa2[k3];
      float *pa = pa0;
      for (int64_t i = 0 ; i < n ; ++i, pa += ps0)
	{
	  float x0 = pa[0], y0 = pa[ps1], z0 = pa[2*ps1];
	  float x = p00*x0 + p01*y0 + p02*z0 + p03;
	  float y = p10*x0 + p11*y0 + p12*z0 + p13;
	  float z = p20*x0 + p21*y0 + p22*z0 + p23;
	  if (j == 0 && i == 0)
	    { xmin = xmax = x; ymin = ymax = y; zmin = zmax = z; }
	  else
	    {
	      if (x < xmin) xmin = x;
	      else if (x > xmax) xmax = x;
	      if (y < ymin) ymin = y;
	      else if (y > ymax) ymax = y;
	      if (z < zmin) zmin = z;
	      else if (z > zmax) zmax = z;
	    }
	}
    }
  xyz_min[0] = xmin; xyz_min[1] = ymin; xyz_min[2] = zmin;
  xyz_max[0] = xmax; xyz_max[1] = ymax; xyz_max[2] = zmax;
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *point_copies_bounds(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points, positions;
  const char *kwlist[] = {"points", "positions", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_array, &positions))
    return NULL;

  if (positions.dimension() != 3)
    return PyErr_Format(PyExc_ValueError, "Positions array is not 3 dimensional, got %d",
			positions.dimension());
  if (positions.size(1) != 3 || positions.size(2) != 4)
    return PyErr_Format(PyExc_ValueError, "Positions array is not of size Nx3x4, got %s",
			positions.size_string().c_str());

  float *xyz_bounds;
  PyObject *bounds = python_float_array(2, 3, &xyz_bounds);
  point_copies_bounding_box(points, positions, xyz_bounds, xyz_bounds+3);
  return bounds;
}
