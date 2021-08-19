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
// Transform points with shift, scale and linear operations.
//
#include <Python.h>			// use PyObject
#include <math.h>			// use sqrtf()

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use call_template_function()
#include "transform.h"

// ----------------------------------------------------------------------------
//
static void scale_and_shift_vertices(FArray &vertex_positions,
				     float xyz_origin[3], float xyz_step[3])
{
  float *xyz = vertex_positions.values();
  int64_t n = vertex_positions.size(0);
  int64_t n3 = 3 * n;
  for (int64_t k = 0 ; k < n3 ; k += 3)
    for (int a = 0 ; a < 3 ; ++a)
      xyz[k+a] = xyz[k+a] * xyz_step[a] + xyz_origin[a];
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *scale_and_shift_vertices(PyObject *, PyObject *args)
{
  FArray varray;
  float origin[3], step[3];
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&O&"),
			parse_writable_float_n3_array, &varray,
			parse_float_3_array, origin,
			parse_float_3_array, step))
    return NULL;

  scale_and_shift_vertices(varray, origin, step);

  return python_none();
}

// ----------------------------------------------------------------------------
//
static void scale_vertices(FArray &vertex_positions, float xyz_step[3])
{
  float *xyz = vertex_positions.values();
  int64_t n = vertex_positions.size(0);
  int64_t n3 = 3 * n;
  for (int64_t k = 0 ; k < n3 ; k += 3)
    for (int a = 0 ; a < 3 ; ++a)
      xyz[k+a] *= xyz_step[a];
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *scale_vertices(PyObject *, PyObject *args)
{
  FArray varray;
  float step[3];
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&"),
			parse_writable_float_n3_array, &varray,
			parse_float_3_array, step))
    return NULL;

  scale_vertices(varray, step);

  return python_none();
}

// ----------------------------------------------------------------------------
//
static void shift_vertices(FArray &vertex_positions, float xyz_origin[3])
{
  float *xyz = vertex_positions.values();
  int64_t n = vertex_positions.size(0);
  int64_t n3 = 3 * n;
  for (int64_t k = 0 ; k < n3 ; k += 3)
    for (int a = 0 ; a < 3 ; ++a)
      xyz[k+a] += xyz_origin[a];
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *shift_vertices(PyObject *, PyObject *args)
{
  FArray varray;
  float origin[3];
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&"),
			parse_writable_float_n3_array, &varray,
			parse_float_3_array, origin))
    return NULL;

  shift_vertices(varray, origin);

  return python_none();
}

// ----------------------------------------------------------------------------
//
static void affine_transform_vertices(FArray &vertex_positions, float tf[3][4])
{
  float *xyz = vertex_positions.values();
  int64_t n = vertex_positions.size(0);
  int64_t s0 = vertex_positions.stride(0), s1 = vertex_positions.stride(1);
  float t00 = tf[0][0], t01 = tf[0][1], t02 = tf[0][2], t03 = tf[0][3];
  float t10 = tf[1][0], t11 = tf[1][1], t12 = tf[1][2], t13 = tf[1][3];
  float t20 = tf[2][0], t21 = tf[2][1], t22 = tf[2][2], t23 = tf[2][3];
  for (int64_t k = 0 ; k < n ; ++k)
    {
      float *px = xyz + s0*k;
      float *py = px + s1;
      float *pz = py + s1;
      float x = *px, y = *py, z = *pz;
      *px = t00*x + t01*y + t02*z + t03;
      *py = t10*x + t11*y + t12*z + t13;
      *pz = t20*x + t21*y + t22*z + t23;
    }
}

// ----------------------------------------------------------------------------
//
static void affine_transform_vertices(DArray &vertex_positions, double tf[3][4])
{
  double *xyz = vertex_positions.values();
  int64_t n = vertex_positions.size(0);
  int64_t s0 = vertex_positions.stride(0), s1 = vertex_positions.stride(1);
  double t00 = tf[0][0], t01 = tf[0][1], t02 = tf[0][2], t03 = tf[0][3];
  double t10 = tf[1][0], t11 = tf[1][1], t12 = tf[1][2], t13 = tf[1][3];
  double t20 = tf[2][0], t21 = tf[2][1], t22 = tf[2][2], t23 = tf[2][3];
  for (int64_t k = 0 ; k < n ; ++k)
    {
      double *px = xyz + s0*k;
      double *py = px + s1;
      double *pz = py + s1;
      double x = *px, y = *py, z = *pz;
      *px = t00*x + t01*y + t02*z + t03;
      *py = t10*x + t11*y + t12*z + t13;
      *pz = t20*x + t21*y + t22*z + t23;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *affine_transform_vertices(PyObject *, PyObject *args)
{
  FArray varray;
  float tf[3][4];
  DArray v64array;
  double tf64[3][4];
  if (PyArg_ParseTuple(args, const_cast<char *>("O&O&"),
		       parse_writable_float_n3_array, &varray,
		       parse_float_3x4_array, tf))
    affine_transform_vertices(varray, tf);
  else if (PyArg_ParseTuple(args, const_cast<char *>("O&O&"),
			    parse_writable_double_n3_array, &v64array,
			    parse_double_3x4_array, tf64))
    {
      PyErr_Clear();
      affine_transform_vertices(v64array, tf64);
    }
  else
    return NULL;


  return python_none();
}

// ----------------------------------------------------------------------------
//
static void affine_transform_normals(FArray &vertex_positions, float tf[3][3])
{
  float *xyz = vertex_positions.values();
  int64_t n = vertex_positions.size(0);
  int64_t s0 = vertex_positions.stride(0), s1 = vertex_positions.stride(1);

  float t00 = tf[0][0], t01 = tf[0][1], t02 = tf[0][2];
  float t10 = tf[1][0], t11 = tf[1][1], t12 = tf[1][2];
  float t20 = tf[2][0], t21 = tf[2][1], t22 = tf[2][2];

  float r00 = t11 * t22 - t12 * t21;
  float r10 = t21 * t02 - t22 * t01;
  float r20 = t01 * t12 - t02 * t11;
  float r01 = t12 * t20 - t10 * t22;
  float r11 = t22 * t00 - t20 * t02;
  float r21 = t02 * t10 - t00 * t12;
  float r02 = t10 * t21 - t11 * t20;
  float r12 = t20 * t01 - t21 * t00;
  float r22 = t00 * t11 - t01 * t10;
  float det = t00 * r00 + t01 * r01 + t02 * r02;
  r00 /= det; r01 /= det; r02 /= det;
  r10 /= det; r11 /= det; r12 /= det;
  r20 /= det; r21 /= det; r22 /= det;

  for (int64_t k = 0 ; k < n ; ++k)
    {
      float *px = xyz + s0*k;
      float *py = px + s1;
      float *pz = py + s1;
      float x = *px, y = *py, z = *pz;
      *px = r00*x + r01*y + r02*z;
      *py = r10*x + r11*y + r12*z;
      *pz = r20*x + r21*y + r22*z;

      float len = sqrtf(*px * *px + *py * *py + *pz * *pz);
      if (len != 0)
	{
	  *px /= len;
	  *py /= len;
	  *pz /= len;
	}
    }
}

// ----------------------------------------------------------------------------
//
static void affine_transform_normals(DArray &vertex_positions, double tf[3][3])
{
  double *xyz = vertex_positions.values();
  int64_t n = vertex_positions.size(0);
  int64_t s0 = vertex_positions.stride(0), s1 = vertex_positions.stride(1);

  double t00 = tf[0][0], t01 = tf[0][1], t02 = tf[0][2];
  double t10 = tf[1][0], t11 = tf[1][1], t12 = tf[1][2];
  double t20 = tf[2][0], t21 = tf[2][1], t22 = tf[2][2];

  double r00 = t11 * t22 - t12 * t21;
  double r10 = t21 * t02 - t22 * t01;
  double r20 = t01 * t12 - t02 * t11;
  double r01 = t12 * t20 - t10 * t22;
  double r11 = t22 * t00 - t20 * t02;
  double r21 = t02 * t10 - t00 * t12;
  double r02 = t10 * t21 - t11 * t20;
  double r12 = t20 * t01 - t21 * t00;
  double r22 = t00 * t11 - t01 * t10;
  double det = t00 * r00 + t01 * r01 + t02 * r02;
  r00 /= det; r01 /= det; r02 /= det;
  r10 /= det; r11 /= det; r12 /= det;
  r20 /= det; r21 /= det; r22 /= det;

  for (int64_t k = 0 ; k < n ; ++k)
    {
      double *px = xyz + s0*k;
      double *py = px + s1;
      double *pz = py + s1;
      double x = *px, y = *py, z = *pz;
      *px = r00*x + r01*y + r02*z;
      *py = r10*x + r11*y + r12*z;
      *pz = r20*x + r21*y + r22*z;

      double len = sqrt(*px * *px + *py * *py + *pz * *pz);
      if (len != 0)
	{
	  *px /= len;
	  *py /= len;
	  *pz /= len;
	}
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *affine_transform_normals(PyObject *, PyObject *args)
{
  FArray varray;
  float tf[3][3];
  DArray v64array;
  double tf64[3][3];
  if (PyArg_ParseTuple(args, const_cast<char *>("O&O&"),
		       parse_writable_float_n3_array, &varray,
		       parse_float_3x3_array, tf))
    affine_transform_normals(varray, tf);
  else if (PyArg_ParseTuple(args, const_cast<char *>("O&O&"),
			    parse_writable_double_n3_array, &v64array,
			    parse_double_3x3_array, tf64))
    {
      PyErr_Clear();
      affine_transform_normals(v64array, tf64);
    }
  else
    return NULL;


  return python_none();
}
