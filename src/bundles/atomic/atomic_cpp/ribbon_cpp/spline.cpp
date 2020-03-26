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

// Need _USE_MATH_DEFINES on Windows to get M_PI from cmath
#define _USE_MATH_DEFINES
#include <cmath>			// use std:isnan()
#include <iostream>

#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use python_float_array
#include "normals.h"			// use parallel_transport_normals, dihedral_angle

// -----------------------------------------------------------------------------
//
static void cubic_path(const double *c, double tmin, double tmax, int n, float *coords, float *tangents)
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
extern "C" const char *cubic_path_doc =
  "cubic_path(coeffs, tmin, tmax, num_points) -> coords, tangents\n"
  "\n"
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

// -----------------------------------------------------------------------------
//
static void spline_path(const double *coeffs, int nseg, const float *normals,
			const unsigned char *flip, const unsigned char *twist, int ndiv,
			float *ca, float *ta, float *na)
{
  int np = ndiv/2;
  cubic_path(coeffs, -0.3, 0, np+1, ca, ta);
  bool backwards = true;  // Parallel transport normal backwards
  parallel_transport(np+1, ta, normals, na, backwards);
  ca += 3*np; ta += 3*np ; na += 3*np;
  
  const float *end_normal = NULL;
  float flipped_normal[3];
  for (int seg = 0 ; seg < nseg ; ++seg)
    {
      np = ndiv+1;
      cubic_path(coeffs+12*seg, 0, 1, np, ca, ta);
      const float *start_normal = (seg == 0 ? normals : end_normal);
      parallel_transport(np, ta, start_normal, na);
      end_normal = normals + 3*(seg + 1);
      
      if (twist[seg])
	{
	  if (flip[seg])
	    {
	      // Decide whether to flip the spline segment start normal so that it aligns
	      // better with the preceding segment parallel transported normal.
	      float a = dihedral_angle(na + 3*ndiv, end_normal, ta + 3*ndiv);
	      bool flip= (fabs(a) > 0.6 * M_PI);	// Not sure why this is not 0.5 * M_PI
	      if (flip)
		{
		  for (int i = 0 ; i < 3 ; ++i)
		    flipped_normal[i] = -end_normal[i];
		  end_normal = flipped_normal;
		}
	    }
	  smooth_twist(ta, np, na, end_normal);
	}
      ca += 3*ndiv; ta += 3*ndiv ; na += 3*ndiv;
    }

  np = (ndiv + 1)/2;
  cubic_path(coeffs + 12*(nseg-1), 1, 1.3, np, ca, ta);
  parallel_transport(np, ta, end_normal, na);
}

// -----------------------------------------------------------------------------
//
extern "C" const char *spline_path_doc =
  "spline_path(coeffs, normals, flip_normals, twist, ndiv) -> coords, tangents, normals\n"
  "\n"
  "Compute a path in 3D from segments (x(t),y(t),z(t)) cubic polynomials in t.\n"
  "The path is also extrapolated before the first segment and after the last segment.\n"
  "Polynomial coefficients are given by N 3x4 matrix coeffs, 64-bit float.\n"
  "Normal vectors a the start of the N segments are specified and are.\n"
  "parallel transported along the path.  If flip_normals[i] is true and\n"
  "the parallel transported normal for segment i does is more than 90 degrees\n"
  "away from the specified normal for segment i+1 then the i+1 segment\n"
  "starts with a flipped normal.  If twist[i] then the segment normals\n"
  "are rotated so that the end of segment normal is in the same plane\n"
  "as the starting normal of the next segment, with each normal in the\n"
  "segment having a twist about its tangent vector applied in linearly\n"
  "increasing amounts.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "coeffs : N by 3 by 4 float64 array\n"
  "  x,y,z cubic polynomial coefficients c0 + c1*t + c2*t*t + c3*t*t*t for N segments.\n"
  "normals : N+1 by 3 float array\n"
  "  normal vectors for segment end points.\n"
  "flip_normals : unsigned char\n"
  "  boolean value for each segment whether to allow flipping normals.\n"
  "twist : unsigned char\n"
  "  boolean value for each segment whether to twist normals.\n"
  "ndiv : int\n"
  "  number of points per segment.  Left end is include, right end excluded.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "coords : M by 3 float array\n"
  "  points on cubic path. M = (N+1) * ndiv\n"
  "tangents : M by 3 float array\n"
  "  normalized tangent vectors at each point of path.\n"
  "normals : M by 3 float array\n"
  "  normal vectors at each point of path.\n";

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *spline_path(PyObject *, PyObject *args, PyObject *keywds)
{
  DArray coeffs;
  FArray normals;
  Reference_Counted_Array::Array<unsigned char> flip, twist; // boolean
  int ndiv;
  const char *kwlist[] = {"coeffs", "normals", "flip_normals", "twist", "ndiv", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&i"),
				   (char **)kwlist,
				   parse_contiguous_double_n34_array, &coeffs,
				   parse_float_n3_array, &normals,
				   parse_uint8_n_array, &flip,
				   parse_uint8_n_array, &twist,
				   &ndiv))
    return NULL;

  if (!normals.is_contiguous() || !flip.is_contiguous() || !twist.is_contiguous())
    {
      PyErr_Format(PyExc_TypeError,
		   "spline_path(): normals, flip and twist arrays must be contiguous");
      return NULL;
    }
  if (coeffs.size(0)+1 != normals.size(0))
    {
      PyErr_Format(PyExc_TypeError,
		   "spline_path(): Normals array (%s) must be one longer than coefficients array (%s)",
		   normals.size_string().c_str(), coeffs.size_string().c_str());
      return NULL;
    }
  if (flip.size(0) < coeffs.size(0) || twist.size(0) < coeffs.size(0))
    {
      PyErr_Format(PyExc_TypeError,
		   "spline_path(): Flip array (%s) and twist array (%s) must have same size as coefficients array (%s)",
		   flip.size_string().c_str(), twist.size_string().c_str(), coeffs.size_string().c_str());
      return NULL;
    }

  int nseg = coeffs.size(0);
  int num_points = (nseg+1) * ndiv;
  float *ca, *ta, *na;
  PyObject *pcoords = python_float_array(num_points, 3, &ca);
  PyObject *ptangents = python_float_array(num_points, 3, &ta);
  PyObject *pnormals = python_float_array(num_points, 3, &na);

  spline_path(coeffs.values(), nseg, normals.values(), flip.values(), twist.values(), ndiv,
	      ca, ta, na);

  PyObject *ctn = python_tuple(pcoords, ptangents, pnormals);
  return ctn;
}
