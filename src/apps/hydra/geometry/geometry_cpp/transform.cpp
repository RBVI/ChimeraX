// ----------------------------------------------------------------------------
// Transform points with shift, scale and linear operations.
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use call_template_function()

// ----------------------------------------------------------------------------
//
static void scale_and_shift_vertices(FArray &vertex_positions,
				     float xyz_origin[3], float xyz_step[3])
{
  float *xyz = vertex_positions.values();
  int n = vertex_positions.size(0);
  int n3 = 3 * n;
  for (int k = 0 ; k < n3 ; k += 3)
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

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static void scale_vertices(FArray &vertex_positions, float xyz_step[3])
{
  float *xyz = vertex_positions.values();
  int n = vertex_positions.size(0);
  int n3 = 3 * n;
  for (int k = 0 ; k < n3 ; k += 3)
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

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static void shift_vertices(FArray &vertex_positions, float xyz_origin[3])
{
  float *xyz = vertex_positions.values();
  int n = vertex_positions.size(0);
  int n3 = 3 * n;
  for (int k = 0 ; k < n3 ; k += 3)
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

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static void affine_transform_vertices(FArray &vertex_positions, float tf[3][4])
{
  float *xyz = vertex_positions.values();
  int n = vertex_positions.size(0);
  int s0 = vertex_positions.stride(0), s1 = vertex_positions.stride(1);
  float t00 = tf[0][0], t01 = tf[0][1], t02 = tf[0][2], t03 = tf[0][3];
  float t10 = tf[1][0], t11 = tf[1][1], t12 = tf[1][2], t13 = tf[1][3];
  float t20 = tf[2][0], t21 = tf[2][1], t22 = tf[2][2], t23 = tf[2][3];
  for (int k = 0 ; k < n ; ++k)
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
  int n = vertex_positions.size(0);
  int s0 = vertex_positions.stride(0), s1 = vertex_positions.stride(1);
  double t00 = tf[0][0], t01 = tf[0][1], t02 = tf[0][2], t03 = tf[0][3];
  double t10 = tf[1][0], t11 = tf[1][1], t12 = tf[1][2], t13 = tf[1][3];
  double t20 = tf[2][0], t21 = tf[2][1], t22 = tf[2][2], t23 = tf[2][3];
  for (int k = 0 ; k < n ; ++k)
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
    affine_transform_vertices(v64array, tf64);
  else
    return NULL;


  Py_INCREF(Py_None);
  return Py_None;
}
