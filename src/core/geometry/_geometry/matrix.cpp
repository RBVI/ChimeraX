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
extern "C" PyObject *multiply_matrices_f64(PyObject *, PyObject *args, PyObject *keywds)
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
    }

  Py_INCREF(py_result);
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
    }

  Py_INCREF(py_result);
  return py_result;
}
