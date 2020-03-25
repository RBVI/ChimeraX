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

#include <arrays/pythonarray.h>		// use parse_uint8_n_array()
#include <arrays/rcarray.h>		// use CArray

inline float inner(const float* u, const float* v)
{
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

inline float norm(const float* u)
{
    return sqrtf(inner(u,u));
}

inline float* cross(const float* u, const float* v, float* result)
{
    result[0] = u[1]*v[2] - u[2]*v[1];
    result[1] = u[2]*v[0] - u[0]*v[2];
    result[2] = u[0]*v[1] - u[1]*v[0];
    return result;
}

inline float dihedral_angle(const float *u, const float *v, const float *t)
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

static void _rotate_around(float* n, float c, float s, float* v)
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

static void _parallel_transport_normals(int num_pts, float* tangents, float* n0, float* normals)
{
    // First normal is same as given normal
    normals[0] = n0[0];
    normals[1] = n0[1];
    normals[2] = n0[2];
    // n: normal updated at each step
    // b: binormal defined by cross product of two consecutive tangents
    // b_hat: normalized b
    float n[3] = { n0[0], n0[1], n0[2] };
    float b[3];
    float b_hat[3];
    for (int i = 1; i != num_pts; ++i) {
        float *ti1 = tangents + (i - 1) * 3;
        float *ti = ti1 + 3;
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
        float *ni = normals + i * 3;
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

static void smooth_twist(float *tangents, int num_pts, float *normals, float *n_end)
{
    // Figure out what twist is needed to make the
    // ribbon end up with the desired ending normal
    float *n = normals + (num_pts - 1) * 3;
    float *t = tangents + (num_pts - 1) * 3;
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
smooth_twist(PyObject *, PyObject *args, PyObject *keywds)
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
parallel_transport(PyObject *, PyObject *args, PyObject *keywds)
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
  _parallel_transport_normals(num_pts, tang.values(), start_normal, normals);

  return py_normals;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
dihedral_angle(PyObject *, PyObject *args, PyObject *keywds)
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
