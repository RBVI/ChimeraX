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

// ----------------------------------------------------------------------------
// Compute matrices
//
#include <Python.h>			// use PyObject
#include <cmath>            // sqrt

#include <arrays/pythonarray.h>		// use array_from_python()
#include "matrix.h"

inline void normalize(double *xyz)
{
    double len = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]);
    if (len == 0.0) return;
    xyz[0] /= len;
    xyz[1] /= len;
    xyz[2] /= len;
}

inline void cross(const double *v1, const double *v2, double *out)
{
    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

static PyObject *look_at(double* from_pt, double* to_pt, double* up)
{
    double x[3], y[3], z[3];
    double diff[3], trans[3];
    for (int i = 0; i < 3; ++i) {
        diff[i] = to_pt[i] - from_pt[i];
    }
    normalize(diff);
    cross(up, diff, x);
    normalize(x);
    cross(diff, x, y);
    normalize(y);
    z[0] = -diff[0]; z[1] = -diff[1]; z[2] = -diff[2];
    auto fx = from_pt[0];
    auto fy = from_pt[1];
    auto fz = from_pt[2];
    trans[0] = 0.0 - (x[0]*fx + x[1]*fy + x[2]*fz);
    trans[1] = 0.0 - (y[0]*fx + y[1]*fy + y[2]*fz);
    trans[2] = 0.0 - (z[0]*fx + z[1]*fy + z[2]*fz);
    double matrix[12];
    double* m = &matrix[0];
    *m++ = x[0]; *m++ = x[1], *m++ = x[2]; *m++ = trans[0];
    *m++ = y[0]; *m++ = y[1], *m++ = y[2]; *m++ = trans[1];
    *m++ = z[0]; *m++ = z[1], *m++ = z[2]; *m = trans[2];
    return c_array_to_python(matrix, 3, 4);
}

extern "C" PyObject *look_at(PyObject *, PyObject *args)
{
  double from_pt[3], to_pt[3], up[3];
  if (PyArg_ParseTuple(args, const_cast<char *>("O&O&O&"),
		       parse_double_3_array, from_pt,
		       parse_double_3_array, to_pt,
		       parse_double_3_array, up))
    {
      return look_at(from_pt, to_pt, up);
    }
  return NULL;
}

// ----------------------------------------------------------------------------
// 3x4 matrix indices
//  0   1   2   3
//  4   5   6   7
//  8   9  10  11
//
static void multiply_matrices(double *m, double *n, double *r)
{
  // Save result in locals in case result matrix is same array as
  // one of the matrices being multiplied.
  double r0 = m[0]*n[0] + m[1]*n[4] + m[2]*n[8];
  double r1 = m[0]*n[1] + m[1]*n[5] + m[2]*n[9];
  double r2 = m[0]*n[2] + m[1]*n[6] + m[2]*n[10];
  double r3 = m[0]*n[3] + m[1]*n[7] + m[2]*n[11] + m[3];

  double r4 = m[4]*n[0] + m[5]*n[4] + m[6]*n[8];
  double r5 = m[4]*n[1] + m[5]*n[5] + m[6]*n[9];
  double r6 = m[4]*n[2] + m[5]*n[6] + m[6]*n[10];
  double r7 = m[4]*n[3] + m[5]*n[7] + m[6]*n[11] + m[7];

  double r8 = m[8]*n[0] + m[9]*n[4] + m[10]*n[8];
  double r9 = m[8]*n[1] + m[9]*n[5] + m[10]*n[9];
  double r10 = m[8]*n[2] + m[9]*n[6] + m[10]*n[10];
  double r11 = m[8]*n[3] + m[9]*n[7] + m[10]*n[11] + m[11];

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
  r[8] = r8; r[9] = r9; r[10] = r10; r[11] = r11;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *multiply_matrices(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_result = NULL;
  DArray m1, m2;
  const char *kwlist[] = {"matrix1", "matrix2", "result", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&|O"),
				   (char **)kwlist,
				   parse_contiguous_double_3x4_array, &m1,
				   parse_contiguous_double_3x4_array, &m2,
				   &py_result))
    return NULL;

  if (py_result == NULL)
    {
      double *r;
      py_result = python_double_array(3, 4, &r);
      multiply_matrices(m1.values(), m2.values(), r);
    }
  else
    {
      DArray result;
      if (!parse_contiguous_double_3x4_array(py_result, &result))
	return NULL;
      multiply_matrices(m1.values(), m2.values(), result.values());
      Py_INCREF(py_result);
    }

  return py_result;
}

// ----------------------------------------------------------------------------
//
static void multiply_matrix_lists(double *m1, int n1, double *m2, int n2, double *r)
{
  for (int i1 = 0 ; i1 < n1 ; ++i1)
    for (int i2 = 0 ; i2 < n2 ; ++i2)
      multiply_matrices(m1 + 12*i1, m2 + 12*i2, r + 12*n2*i1 + 12*i2);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *multiply_matrix_lists(PyObject *, PyObject *args, PyObject *keywds)
{
  DArray m1, m2;
  int n1, n2;
  PyObject *py_result = NULL;
  const char *kwlist[] = {"matrices1", "n1", "matrices2", "n2", "result", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&iO&i|O"),
				   (char **)kwlist,
				   parse_contiguous_double_n34_array, &m1,
				   &n1,
				   parse_contiguous_double_n34_array, &m2,
				   &n2,
				   &py_result))
    return NULL;

  if (py_result == NULL)
    {
      double *r;
      py_result = python_double_array(n1*n2, 3, 4, &r);
      multiply_matrix_lists(m1.values(), n1, m2.values(), n2, r);
    }
  else
    {
      DArray result;
      if (!parse_contiguous_double_n34_array(py_result, &result))
	return NULL;
      if (result.size(0) != n1*n2)
	{
	  PyErr_Format(PyExc_TypeError,
		       "Require result array size %d x 3 x 4, got %s by 3 by 4",
		       n1*n2, result.size_string(0).c_str());
	  return NULL;
	}
      multiply_matrix_lists(m1.values(), n1, m2.values(), n2, result.values());
      Py_INCREF(py_result);
    }

  return py_result;
}

// ----------------------------------------------------------------------------
// 3x4 matrix indices
//  0   1   2   3
//  4   5   6   7
//  8   9  10  11
//
static void opengl_matrix(double *m, float *r)
{
  // Save result in locals in case result matrix is same array as
  // one of the matrices being multiplied.
  r[0] = (float)m[0];
  r[1] = (float)m[4];
  r[2] = (float)m[8];
  r[3] = 0;
  r[4] = (float)m[1];
  r[5] = (float)m[5];
  r[6] = (float)m[9];
  r[7] = 0;
  r[8] = (float)m[2];
  r[9] = (float)m[6];
  r[10] = (float)m[10];
  r[11] = 0;
  r[12] = (float)m[3];
  r[13] = (float)m[7];
  r[14] = (float)m[11];
  r[15] = 1;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *opengl_matrix(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_result = NULL;
  DArray m;
  const char *kwlist[] = {"matrix_3x4", "result", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&|O"),
				   (char **)kwlist,
				   parse_contiguous_double_3x4_array, &m,
				   &py_result))
    return NULL;

  if (py_result == NULL)
    {
      float *r;
      py_result = python_float_array(4, 4, &r);
      opengl_matrix(m.values(), r);
    }
  else
    {
      FArray result;
      if (!parse_contiguous_float_4x4_array(py_result, &result))
	return NULL;
      opengl_matrix(m.values(), result.values());
      Py_INCREF(py_result);
    }

  return py_result;
}

// ----------------------------------------------------------------------------
//
static void opengl_matrices(double *m34, int64_t n, float *m44)
{
  for (int64_t i = 0 ; i < n ; ++i)
    opengl_matrix(m34 + 12*i, m44 + 16*i);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *opengl_matrices(PyObject *, PyObject *args, PyObject *keywds)
{
  DArray m;
  int n;
  PyObject *py_result = NULL;
  const char *kwlist[] = {"matrices", "n", "result", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&i|O"),
				   (char **)kwlist,
				   parse_contiguous_double_n34_array, &m,
				   &n,
				   &py_result))
    return NULL;

  if (py_result == NULL)
    {
      float *r;
      py_result = python_float_array(n, 4, 4, &r);
      opengl_matrices(m.values(), n, r);
    }
  else
    {
      FArray result;
      if (!parse_contiguous_float_n44_array(py_result, &result))
	return NULL;
      if (result.size(0) != n)
	{
	  PyErr_Format(PyExc_TypeError,
		       "Require result array size %d x 4 x 4, got %s by 4 by 4",
		       n, result.size_string(0).c_str());
	  return NULL;
	}
      opengl_matrices(m.values(), n, result.values());
      Py_INCREF(py_result);
    }

  return py_result;
}

// ----------------------------------------------------------------------------
//
static bool same_matrix(double *m, double *n)
{
  for (int i = 0 ; i < 12 ; ++i)
    if (m[i] != n[i])
      return false;
  return true;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *same_matrix(PyObject *, PyObject *args, PyObject *keywds)
{
  DArray m1, m2;
  const char *kwlist[] = {"matrix1", "matrix2", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_contiguous_double_3x4_array, &m1,
				   parse_contiguous_double_3x4_array, &m2))
    return NULL;

  bool same = same_matrix(m1.values(), m2.values());
  return python_bool(same);
}

// ----------------------------------------------------------------------------
//
static bool is_identity_matrix(double *m, double tolerance)
{
  bool id;
  if (tolerance == 0)
    id = (m[0] == 1 && m[1] == 0 && m[2] == 0 && m[3] == 0 &&
	  m[4] == 0 && m[5] == 1 && m[6] == 0 && m[7] == 0 &&
	  m[8] == 0 && m[9] == 0 && m[10] == 1 && m[11] == 0);
  else
    {
      double t = tolerance;
      id = (fabs(m[0]-1) <= t && fabs(m[1]) <= t && fabs(m[2]) <= t && fabs(m[3]) <= t &&
	    fabs(m[4]) <= t && fabs(m[5]-1) <= t && fabs(m[6]) <= t && fabs(m[7]) <= t &&
	    fabs(m[8]) <= t && fabs(m[9]) <= t && fabs(m[10]-1) <= t && fabs(m[11]) <= t);
    }
  return id;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *is_identity_matrix(PyObject *, PyObject *args, PyObject *keywds)
{
  DArray m;
  double tolerance = 0;
  const char *kwlist[] = {"matrix", "tolerance", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&|d"),
				   (char **)kwlist,
				   parse_contiguous_double_3x4_array, &m,
				   &tolerance))
    return NULL;

  bool id = is_identity_matrix(m.values(), tolerance);
  return python_bool(id);
}

// ----------------------------------------------------------------------------
//
static void invert_orthonormal(double *m, double *result)
{
  double *r = result;
  double s0 = m[0]*m[3] + m[4]*m[7] + m[8]*m[11];
  double s1 = m[1]*m[3] + m[5]*m[7] + m[9]*m[11];
  double s2 = m[2]*m[3] + m[6]*m[7] + m[10]*m[11];
  // Use temporaries in case result is the same array as m, ie inverting in place.
  double m01 = m[1], m02 = m[2], m12 = m[6];
  // 3x3 transposed is the inverse.
  r[0] = m[0]; r[1] = m[4]; r[2] = m[8]; r[3] = -s0;
  r[4] = m01; r[5] = m[5]; r[6] = m[9]; r[7] = -s1;
  r[8] = m02; r[9] = m12; r[10] = m[10]; r[11] = -s2;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *invert_orthonormal(PyObject *, PyObject *args, PyObject *keywds)
{
  DArray m;
  PyObject *py_result = NULL;
  const char *kwlist[] = {"matrix", "result", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&|O"),
				   (char **)kwlist,
				   parse_contiguous_double_3x4_array, &m,
				   &py_result))
    return NULL;

  if (py_result == NULL)
    {
      double *r;
      py_result = python_double_array(3, 4, &r);
      invert_orthonormal(m.values(), r);
    }
  else
    {
      DArray result;
      if (!parse_contiguous_double_3x4_array(py_result, &result))
	return NULL;
      invert_orthonormal(m.values(), result.values());
      Py_INCREF(py_result);
    }

  return py_result;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *set_translation_matrix(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *result;
  double t[3];
  const char *kwlist[] = {"translation", "matrix", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O"),
				   (char **)kwlist,
				   parse_double_3_array, &(t[0]),
				   &result))
    return NULL;

  DArray r;
  if (!parse_contiguous_double_3x4_array(result, &r))
	return NULL;

  double *m = r.values();
  m[0] = 1; m[1] = 0; m[2] = 0; m[3] = t[0];
  m[4] = 0; m[5] = 1; m[6] = 0; m[7] = t[1];
  m[8] = 0; m[9] = 0; m[10] = 1; m[11] = t[2];

  return python_none();
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *set_scale_matrix(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *result;
  double s[3];
  const char *kwlist[] = {"scale", "matrix", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O"),
				   (char **)kwlist,
				   parse_double_3_array, &(s[0]),
				   &result))
    return NULL;

  DArray r;
  if (!parse_contiguous_double_3x4_array(result, &r))
	return NULL;

  double *m = r.values();
  m[0] = s[0]; m[1] = 0; m[2] = 0; m[3] = 0;
  m[4] = 0; m[5] = s[1]; m[6] = 0; m[7] = 0;
  m[8] = 0; m[9] = 0; m[10] = s[2]; m[11] = 0;

  return python_none();
}
