// ----------------------------------------------------------------------------
// Compute a contour surface.
//
#include <Python.h>			// use PyObject

// #include <iostream>			// use std:cerr for debugging
#include <stdexcept>			// use std::runtime_error

#include "contour.h"			// use surface()
#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use call_template_function()

using namespace Contour_Calculation;

// ----------------------------------------------------------------------------
//
template <class T>
void contour_surface(const Reference_Counted_Array::Array<T> &data,
		     float threshold, bool cap_faces, Contour_Surface **cs)
{
  // contouring calculation requires contiguous array
  // put sizes in x, y, z order
  Index size[3] = {data.size(2), data.size(1), data.size(0)};
  Stride stride[3] = {data.stride(2), data.stride(1), data.stride(0)};
  *cs = surface(data.values(), size, stride, threshold, cap_faces);
}

// ----------------------------------------------------------------------------
//
static PyObject *surface_py2(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_data;
  float threshold;
  int cap_faces = 1, return_normals = 0;
  const char *kwlist[] = {"data", "threshold", "cap_faces", "calculate_normals", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("Of|ii"),
				   (char **)kwlist,
				   &py_data, &threshold, &cap_faces,
				   &return_normals))
    return NULL;

  
  Numeric_Array data;
  try
    {
      data = array_from_python(py_data, 3);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  Contour_Surface *cs;
  Py_BEGIN_ALLOW_THREADS
  call_template_function(contour_surface, data.value_type(),
  			 (data, threshold, cap_faces, &cs));
  Py_END_ALLOW_THREADS

  float *vxyz;
  int *tvi;
  PyObject *vertex_xyz = python_float_array(cs->vertex_count(), 3, &vxyz);
  PyObject *tv_indices = python_int_array(cs->triangle_count(), 3, &tvi);
  cs->geometry(vxyz, reinterpret_cast<Index *>(tvi));

  PyObject *geom = PyTuple_New(return_normals ? 3 : 2);
  PyTuple_SetItem(geom, 0, vertex_xyz);
  PyTuple_SetItem(geom, 1, tv_indices);

  if (return_normals)
    {
      float *nxyz;
      PyObject *normals = python_float_array(cs->vertex_count(), 3, &nxyz);
      cs->normals(nxyz);
      PyTuple_SetItem(geom, 2, normals);
    }
  delete cs;

  return geom;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *surface_py(PyObject *s, PyObject *args, PyObject *keywds)
{
  try
    {
      return surface_py2(s, args, keywds);
    }
  catch (std::bad_alloc&)
    {
      PyErr_SetString(PyExc_MemoryError, "Out of memory");
      return NULL;
    }
  catch (std::runtime_error &e)
    {
      // pythonarray.cpp throws runtime error on allocation failure.
      PyErr_SetString(PyExc_MemoryError, e.what());
      return NULL;
    }
}

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

// ----------------------------------------------------------------------------
// Swap vertex 1 and 2 of each triangle.
//
static void reverse_triangle_vertex_order(IArray &triangles)
{
  int *ta = triangles.values();
  int n = triangles.size(0);
  int s0 = triangles.stride(0), s1 = triangles.stride(1);
  for (int t = 0 ; t < n ; ++t)
    {
      int i1 = s0*t+s1, i2 = i1 + s1;
      int v1 = ta[i1], v2 = ta[i2];
      ta[i1] = v2;
      ta[i2] = v1;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *reverse_triangle_vertex_order(PyObject *, PyObject *args)
{
  IArray tarray;
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&"),
			parse_writable_int_n3_array, &tarray))
    return NULL;

  reverse_triangle_vertex_order(tarray);

  Py_INCREF(Py_None);
  return Py_None;
}
