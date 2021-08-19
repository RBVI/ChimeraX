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

#include <arrays/pythonarray.h>		// use parse_uint8_n_array()
#include <arrays/rcarray.h>		// use FArray

inline float inner(const float* u, const float* v)
{
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

inline float norm(const float* u)
{
    return sqrtf(inner(u,u));
}

inline void scale_vector(float *v, float s)
{
  v[0] *= s;
  v[1] *= s;
  v[2] *= s;
}

inline void normalize_vector(float* u)
{
  float d = norm(u);
  if (d > 0)
    scale_vector(u, 1/d);
}

inline float* cross(const float* u, const float* v, float* result)
{
    result[0] = u[1]*v[2] - u[2]*v[1];
    result[1] = u[2]*v[0] - u[0]*v[2];
    result[2] = u[0]*v[1] - u[1]*v[0];
    return result;
}

inline void copy_vector(const float *u, float *v)
{
  v[0] = u[0];
  v[1] = u[1];
  v[2] = u[2];
}

inline void zero_vector(float *v)
{
  v[0] = 0;
  v[1] = 0;
  v[2] = 0;
}

inline void subtract_vectors(const float *u, const float *v, float *umv)
{
  umv[0] = u[0] - v[0];
  umv[1] = u[1] - v[1];
  umv[2] = u[2] - v[2];
}

inline void linear_combine_vectors(float a, const float *u, float b, const float *v, float *aubv)
{
  aubv[0] = a*u[0] + b*v[0];
  aubv[1] = a*u[1] + b*v[1];
  aubv[2] = a*u[2] + b*v[2];
}

inline void orthogonal_component(const float *v, const float *u, float *vo)
{
  float d = inner(v, u);
  float u_len = norm(u);
  float f = (u_len > 0 ? -d / u_len : 0);
  for (int i = 0 ; i < 3 ; ++i)
    vo[i] = v[i] + u[i] * f;
}

inline void normal_vector(const float *v, float *n)
{
  if (v[0] != 0 || v[1] != 0)
    { n[0] = -v[1]; n[1] = v[0]; n[2] = v[2]; }
  else
    { n[0] = -v[2]; n[1] = v[1]; n[2] = v[0]; }
}

float dihedral_angle(const float *u, const float *v, const float *t)
{
  float txu[3], txv[3], txtxu[3];
  cross(t, u, txu);
  cross(t, txu, txtxu);
  cross(t, v, txv);
  float x = inner(txu, txv) * norm(t), y = inner(txtxu, txv);
  float a = atan2(y,x);
  return a;
}

// -------------------------------------------------------------------------
// ribbon functions

static void _rotate_around(const float* n, float c, float s, float* v)
{
    float c1 = 1 - c;
    float m00 = c + n[0] * n[0] * c1;
    float m01 = n[0] * n[1] * c1 - s * n[2];
    float m02 = n[2] * n[0] * c1 + s * n[1];
    float m10 = n[0] * n[1] * c1 + s * n[2];
    float m11 = c + n[1] * n[1] * c1;
    float m12 = n[2] * n[1] * c1 - s * n[0];
    float m20 = n[0] * n[2] * c1 - s * n[1];
    float m21 = n[1] * n[2] * c1 + s * n[0];
    float m22 = c + n[2] * n[2] * c1;
    // Use temporary so that v[0] does not get set too soon
    float x = m00 * v[0] + m01 * v[1] + m02 * v[2];
    float y = m10 * v[0] + m11 * v[1] + m12 * v[2];
    float z = m20 * v[0] + m21 * v[1] + m22 * v[2];
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

void parallel_transport(int num_pts, const float* tangents, const float* n0,
			float* normals, bool backwards = false)
{
    // n: normal updated at each step
    // b: binormal defined by cross product of two consecutive tangents
    // b_hat: normalized b
    float n[3] = { n0[0], n0[1], n0[2] };
    float b[3];
    float b_hat[3];
    int istart = 0, iend = num_pts, istep = 1;
    if (backwards)
      { istart = num_pts-1; iend = -1; istep = -1; }
    // First normal is same as given normal
    float *ni = normals + istart * 3;
    ni[0] = n0[0];
    ni[1] = n0[1];
    ni[2] = n0[2];
    for (int i = istart+istep; i != iend; i += istep) {
        const float *ti1 = tangents + (i - istep) * 3;
        const float *ti = ti1 + 3 * istep;
        cross(ti1, ti, b);
        float b_len = sqrtf(inner(b, b));
        if (!std::isnan(b_len) && b_len > 0) {
            b_hat[0] = b[0] / b_len;
            b_hat[1] = b[1] / b_len;
            b_hat[2] = b[2] / b_len;
            float c = inner(ti1, ti);
            if (!std::isnan(c)) {
                float s = sqrtf(1 - c*c);
                if (!std::isnan(s))
                    _rotate_around(b_hat, c, s, n);
            }
        }
        ni = normals + i * 3;
        ni[0] = n[0];
        ni[1] = n[1];
        ni[2] = n[2];
    }
}

inline float delta_to_angle(float twist, float f)
{
    // twist is total twist
    // f is between 0 and 1
    // linear interpolation - show cusp artifact
    // return twist * f;
    // cosine interpolation - second degree continuity
    // return (1 - cos(f * M_PI)) / 2 * twist;
    // sigmoidal interpolation - second degree continuity
    return (1.0 / (1 + exp(-8.0 * (f - 0.5)))) * twist;
}

void smooth_twist(const float *tangents, int num_pts, float *normals, const float *n_end)
{
    // Figure out what twist is needed to make the
    // ribbon end up with the desired ending normal
    float *n = normals + (num_pts - 1) * 3;
    const float *t = tangents + (num_pts - 1) * 3;
    float twist = dihedral_angle(n, n_end, t);

    // Compute fraction per step
    float delta = 1.0 / (num_pts - 1);

    // Apply twist to each normal along path
    for (int i = 1; i != num_pts; ++i) {
        int offset = i * 3;
        float angle = delta_to_angle(twist, i * delta);
        float c = cos(angle);
        float s = sin(angle);
        _rotate_around(tangents + offset, c, s, normals + offset);
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
smooth_twist_py(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray tangents, normals;
  float end_normal[3];
  const char *kwlist[] = {"tangents", "normals", "end_normal", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &tangents,
				   parse_writable_float_n3_array, &normals,
				   parse_float_3_array, &end_normal[0]))

    return NULL;

  FArray tang = tangents.contiguous_array();
  int num_pts = tang.size(0);
  smooth_twist(tang.values(), num_pts, normals.values(), &end_normal[0]);

  return python_none();
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
parallel_transport_py(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray tangents;
  float start_normal[3];
  const char *kwlist[] = {"tangents", "start_normal", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &tangents,
				   parse_float_3_array, &start_normal[0]))
    return NULL;

  FArray tang = tangents.contiguous_array();
  float *normals = NULL;
  int num_pts = tang.size(0);
  PyObject *py_normals = python_float_array(num_pts, 3, &normals);
  parallel_transport(num_pts, tang.values(), start_normal, normals);

  return py_normals;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
dihedral_angle_py(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray tangents;
  float u[3], v[3], t[4];
  const char *kwlist[] = {"u", "v", "t", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&"),
				   (char **)kwlist,
				   parse_float_3_array, &u[0],
      				   parse_float_3_array, &v[0],
				   parse_float_3_array, &t[0]))
    return NULL;

  float a = dihedral_angle(u,v,t);
  return PyFloat_FromDouble(a);
}

// ----------------------------------------------------------------------------
//
static int next_zero_vector(int start, const float *normals, int n)
{
  for (int i = start ; i < n ; ++i)
    if (normals[3*i] == 0 && normals[3*i+1] == 0 && normals[3*i+2] == 0)
      return i;
  return n;
}

// ----------------------------------------------------------------------------
//
static int next_nonzero_vector(int start, const float *normals, int n)
{
  for (int i = start ; i < n ; ++i)
    if ( !(normals[3*i] == 0 && normals[3*i+1] == 0 && normals[3*i+2] == 0))
      return i;
  return n;
}

// ----------------------------------------------------------------------------
//
static void replace_zero_normals(float *normals, int num_pts, const float *tangents)
{
  // Some normal vector has zero length.  Replace it using nearby non-zero normals.
  int s = 0;
  while (true)
    {
      s = next_zero_vector(s, normals, num_pts);
      if (s == num_pts)
	break;
      int e = next_nonzero_vector(s+1, normals, num_pts);
      if (s == 0 && e < num_pts)
	// Set leading 0 normals to first non-zero normal.
	for (int i = 0 ; i < e ; ++i)
	  copy_vector(normals+3*e, normals+3*i);
      else if (s > 0 && e < num_pts)
	{
	  // Linearly interpolate between non-zero normals.
	  float *n0 = normals + 3*(s-1), *n1 = normals + 3*e;
	  for (int i = s ; i < e ; ++i)
	    {
	      float f = (i - (s-1)) / (e - (s-1));
	      linear_combine_vectors(1-f, n0, f, n1, normals + 3*i);
	    }
	}
      else if (s > 0 && e == num_pts)
	// Set trailing 0 normals to last non-zero normal.
	for (int i = s ; i < num_pts ; ++i)
	  copy_vector(normals + 3*(s-1), normals + 3*i);
      else if (s == 0 && e == num_pts)
	{
	  // All normals have zero length, possibly straight line path.
	  for (int i = 0 ; i < num_pts ; ++i)
	    normal_vector(tangents + 3*i, normals + 3*i);
	}
      // Make new normals orthogonal to tangents and unit length.
      for (int i = s ; i < e ; ++i)
	{
	  float *n = normals + 3*i;
	  orthogonal_component(n, tangents +3*i, n);
	  float d = norm(n);
	  if (d > 0)
	    scale_vector(n, 1/d);
	  else
	    {
	      normal_vector(tangents + 3*i, n);
	      scale_vector(n, 1/norm(n));
	    }
	}
      s = e+1;
    }
}

// ----------------------------------------------------------------------------
//  Compute normal vectors to a path perpendicular to tangent vectors.
//  The normal at a path point is obtained by taking the vector perpendicular
//  to the segments to the preceeding and next points and taking its orthogonal
//  component to the tangent vector.  If a normal at a point points opposite
//  the normal of the preceding point (inner product < 0) then it is flipped.
//  If a normal vector is zero for example due to 3 colinear path points
//  then the normal is interpolated from the preceeding and following
//  non-zero normals and orthogonal component to the tangent is taken.
//  For leading or trailing zero normals the nearest non-zero normal is used
//  and orthogonalized against the tangents.  If the whole path is straight
//  or all normals are zero then arbitrary normals perpendicular to the
//  tangents are used.
//
static void path_plane_normals(const float *coords, int num_pts, const float *tangents,
			       float *normals)
{
  //
  // Compute normals by cross-product of vectors to prev and next control points.
  // End points are same as second from end-points.
  //
  float step0[3], step1[3];
  for (int i = 1 ; i < num_pts-1 ; ++i)
    {
      subtract_vectors(coords + 3*(i-1), coords + 3*i, step0);
      subtract_vectors(coords + 3*(i+1), coords + 3*i, step1);
      cross(step0, step1, normals + 3*i);
    }
  copy_vector(normals+3, normals);
  copy_vector(normals + 3*(num_pts-2), normals + 3*(num_pts-1));

  // Take component perpendicular to tangent vectors and make it unit length.
  int num_zero = 0;
  float *prev_normal = NULL;
  for (int i = 0 ; i < num_pts ; ++i)
    {
      float *n = normals + 3*i;
      orthogonal_component(n, tangents + 3*i, n);
      float d = norm(n);
      if (d > 0)
	{
	  if (prev_normal && inner(n, prev_normal) < 0)
	    d = -d;   // Flip this normal to align better with previous one.
	  scale_vector(n, 1/d);
	  prev_normal = n;
	}
      else
	{
	  zero_vector(n);
	  num_zero += 1;
	}
    }

  if (num_zero > 0)
    replace_zero_normals(normals, num_pts, tangents);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
path_plane_normals(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray path, tangents;
  const char *kwlist[] = {"path", "tangents", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &path,
				   parse_float_n3_array, &tangents))
    return NULL;

  if (!path.is_contiguous() || !tangents.is_contiguous())
    {
      PyErr_Format(PyExc_TypeError, "path_plane_normals(): Array arguments must be contiguous.");
      return NULL;
    }
  int num_pts = path.size(0);
  float *normals = NULL;
  PyObject *py_normals = python_float_array(num_pts, 3, &normals);
  path_plane_normals(path.values(), num_pts, tangents.values(), normals);

  return py_normals;
}

// ----------------------------------------------------------------------------
//
static void path_guide_normals(const float *coords, int num_pts,
			       const float *guides, const float *tangents,
			       float *normals)
{
  for (int i = 0 ; i < num_pts ; ++i)
    {
      const float *g = guides + 3*i, *c = coords + 3*i, *t = tangents + 3*i;
      float *n = normals + 3*i;
      subtract_vectors(g, c, n);
      orthogonal_component(n, t, n);
      normalize_vector(n);
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
path_guide_normals(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray path, guides, tangents;
  const char *kwlist[] = {"path", "tangents", "guides", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &path,
				   parse_float_n3_array, &guides,
				   parse_float_n3_array, &tangents))
    return NULL;

  if (!path.is_contiguous() || !guides.is_contiguous() || !tangents.is_contiguous())
    {
      PyErr_Format(PyExc_TypeError, "path_guide_normals(): Array arguments must be contiguous.");
      return NULL;
    }

  int num_pts = path.size(0);
  float *normals = NULL;
  PyObject *py_normals = python_float_array(num_pts, 3, &normals);
  path_guide_normals(path.values(), num_pts, guides.values(), tangents.values(), normals);

  return py_normals;
}
