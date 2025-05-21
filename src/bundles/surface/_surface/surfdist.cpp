// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

#include <Python.h>			// use PyObject
#include <float.h>			// use FLT_MAX
#include <math.h>			// use sqrt()

#include <stdexcept>			// use std::runtime_error

#include <arrays/pythonarray.h>	// use array_from_python()
#include <arrays/rcarray.h>	// use IArray

// ----------------------------------------------------------------------------
//
static void surface_distances(const FArray &xyz, const FArray &varray,
			      const IArray &tarray, float *dist);
static void triangle_distance(const float *xyz,
			      const float *va, const float *vb, const float *vc,
			      float *dist);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *surface_distance(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points, vertex_array, distances;
  IArray triangle_array;
  const char *kwlist[] = {"points", "vertices", "triangles", "distances", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&|O&"), (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_n3_array, &vertex_array,
				   parse_int_n3_array, &triangle_array,
				   parse_writable_float_2d_array, &distances))
    return NULL;

  bool make_distances = (distances.dimension() == 0);
  if (make_distances)
    {
      float *d;
      int n = points.size(0);
      parse_writable_float_2d_array(python_float_array(n,5,&d), &distances);
      for (int k = 0 ; k < n ; ++k)
	d[5*k] = FLT_MAX;
    }      
  else if (distances.size(0) != points.size(0))
    {
      PyErr_SetString(PyExc_TypeError, "distances array must have same size as points array");
      return NULL;
    }
  else if (distances.size(1) != 5)
    {
      PyErr_SetString(PyExc_TypeError, "distances array second dimension must be 5");
      return NULL;
    }
  else if (!distances.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "distances array must be contiguous");
      return NULL;
    }

  surface_distances(points, vertex_array, triangle_array, distances.values());

  PyObject *py_distances = array_python_source(distances, !make_distances);

  return py_distances;
}

// ----------------------------------------------------------------------------
//
inline void copy(const float *u, int ustride, float *v)
  { v[0] = u[0]; v[1] = u[ustride]; v[2] = u[2*ustride]; }

// ----------------------------------------------------------------------------
//
static void surface_distances(const FArray &xyz, const FArray &varray,
			      const IArray &tarray, float *dist)
{
  int np = xyz.size(0), sx0 = xyz.stride(0), sx1 = xyz.stride(1);
  const float *x = xyz.values();
  int sv0 = varray.stride(0), sv1 = varray.stride(1);
  const float *v = varray.values();
  int nt = tarray.size(0), st0 = tarray.stride(0), st1 = tarray.stride(1);
  const int *vi = tarray.values();
  float va[3], vb[3], vc[3], xp[3];
  for (int t = 0 ; t < nt ; ++t)
    {
      int ia = vi[st0*t], ib = vi[st0*t+st1], ic = vi[st0*t+2*st1];
      copy(v+sv0*ia, sv1, va);
      copy(v+sv0*ib, sv1, vb);
      copy(v+sv0*ic, sv1, vc);
      for (int p = 0 ; p < np ; ++p)
	{
	  copy(x+sx0*p, sx1, xp);
	  triangle_distance(xp, va, vb, vc, dist + 5*p);
	}
    }
}

// ----------------------------------------------------------------------------
//
inline void copy(const float *u, float *v)
  { v[0] = u[0]; v[1] = u[1]; v[2] = u[2]; }
inline void add(const float *u, const float *v, float *uv)
  { uv[0] = u[0]+v[0];  uv[1] = u[1]+v[1];  uv[2] = u[2]+v[2]; }
inline void subtract(const float *u, const float *v, float *uv)
  { uv[0] = u[0]-v[0];  uv[1] = u[1]-v[1];  uv[2] = u[2]-v[2]; }
inline void cross(const float *u, const float *v, float *uv)
  { uv[0] = u[1]*v[2]-u[2]*v[1]; uv[1] = u[2]*v[0]-u[0]*v[2]; uv[2] = u[0]*v[1]-u[1]*v[0]; }
inline float cross_dot(const float *u, const float *v, const float *w)
  { return ((u[1]*v[2]-u[2]*v[1])*w[0] + 
	    (u[2]*v[0]-u[0]*v[2])*w[1] +
	    (u[0]*v[1]-u[1]*v[0])*w[2]); }
inline void scale(float a, const float *u, float *au)
  { au[0] = a*u[0]; au[1] = a*u[1]; au[2] = a*u[2]; }
inline float dot(const float *u, const float *v)
  { return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]; }
inline float distance(const float *u, const float *v)
  { float d0 = u[0]-v[0], d1 = u[1]-v[1], d2 = u[2]-v[2];
    return sqrt(d0*d0 + d1*d1 + d2*d2); }
inline float normalize(float *u)
  { float n = sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    if (n > 0) { u[0] /= n; u[1] /= n; u[2] /= n; }
    return n; }

// ----------------------------------------------------------------------------
//
inline int edge_distance(const float *xyz, const float *v1, const float *v2,
			  float *d, float *sxyz)
{
  float v12[3], v[3], exyz[3];
  subtract(v2, v1, v12);
  float v12n2 = dot(v12,v12);
  subtract(xyz, v1, v);
  float vv12 = dot(v,v12);
  if (vv12 >= 0 && vv12 <= v12n2)
    {
      scale(vv12/v12n2, v12, v12);
      add(v1, v12, exyz);
      float de = distance(xyz, exyz);
      if (de < *d)
	{
	  *d = de;
	  copy(exyz, sxyz);
	  return 1;
	}
    }
  return 0;
}

// ----------------------------------------------------------------------------
//
static void triangle_distance(const float *xyz,
			      const float *va, const float *vb, const float *vc,
			      float *dist)
{
  float vab[3], vac[3], vbc[3], n[3], v[3], nvn[3], vt[3], vtvab[3];
  float d = FLT_MAX, sxyz[3] = {0,0,0};
  subtract(vb, va, vab);
  subtract(vc, va, vac);
  subtract(vc, vb, vbc);
  cross(vab, vac, n);
  if (normalize(n) == 0)
    return;	// degenerate triangle

  subtract(xyz, va, v);
  float nv = dot(n, v);
  scale(nv, n, nvn);
  subtract(v, nvn, vt);
  if (cross_dot(n, vab, vt) >= 0 &&
      cross_dot(vac, n, vt) >= 0 &&
      (subtract(vt, vab, vtvab), cross_dot(n, vbc, vtvab)) >= 0)
    {
      // Point projects to triangle interior.
      add(va, vt, sxyz);
      d = distance(xyz, sxyz);
    }
  else
    {
      // Point projects onto an edge.
      edge_distance(xyz, va, vb, &d, sxyz);
      edge_distance(xyz, va, vc, &d, sxyz);
      edge_distance(xyz, vb, vc, &d, sxyz);
      // Point projects to triangle vertex.
      float da = distance(xyz,va), db = distance(xyz,vb), dc = distance(xyz,vc);
      if (da < db && da < dc) { if (da < d) { d = da; copy(va, sxyz); } }
      else if (db < dc)       { if (db < d) { d = db; copy(vb, sxyz); } }
      else                    { if (dc < d) { d = dc; copy(vc, sxyz); } }
    }

  // Update shortest distance
  if (d < dist[0])
    {
      dist[0] = d;
      dist[1] = sxyz[0]; dist[2] = sxyz[1]; dist[3] = sxyz[2];
      dist[4] = (nv >= 0 ? 1.0 : -1.0);
    }
}
