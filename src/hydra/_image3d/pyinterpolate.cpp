// ----------------------------------------------------------------------------
// Python interface to interpolation routines.
//
#include <Python.h>			// use PyObject

#include <stdexcept>			// use std::runtime_error
#include <sstream>			// use std::ostringstream
#include <vector>			// use std::vector

#include "interpolate.h"		// use interpolate_volume_data()
#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// Numeric_Array

using Reference_Counted_Array::Numeric_Array;

typedef Reference_Counted_Array::Array<int> IArray;
typedef Reference_Counted_Array::Array<float> FArray;

// ----------------------------------------------------------------------------
//
static Interpolate::Interpolation_Method interpolation_method(const char *mname);
static void check_array_size(FArray &a, int n, int m,
			     bool require_contiguous = false);
static void check_array_size(FArray &a, int n,
			     bool require_contiguous = false);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *interpolate_volume_data(PyObject *, PyObject *args)
{
  PyObject *py_vertices, *py_vtransform, *py_data, *py_values = NULL;
  const char *method;
  if (!PyArg_ParseTuple(args, const_cast<char *>("OOOs|O"),
			&py_vertices, &py_vtransform, &py_data,	&method,
			&py_values))
    return NULL;

  PyObject *py_outside;
  try
    {
      FArray vertices = array_from_python(py_vertices, 2,
					  Numeric_Array::Float);
      int n = vertices.size(0);
      check_array_size(vertices, n, 3);
      FArray vcontig = vertices.contiguous_array();
      float (*varray)[3] = reinterpret_cast<float(*)[3]>(vcontig.values());

      float vtransform[3][4];
      float *vt = static_cast<float*>(&vtransform[0][0]);
      python_array_to_c(py_vtransform, vt, 3, 4);

      Numeric_Array data = array_from_python(py_data, 3);

      Interpolate::Interpolation_Method m = interpolation_method(method);

      if (py_values == NULL)
	py_values = python_float_array(n);
      else
	Py_INCREF(py_values);

      bool allow_data_copy = false;
      FArray va = array_from_python(py_values, 1, Numeric_Array::Float,
				    allow_data_copy);
      
      check_array_size(va, n, true);

      std::vector<int> outside;

      Py_BEGIN_ALLOW_THREADS
      Interpolate::interpolate_volume_data(varray, n, vtransform, data, m,
					   va.values(), outside);
      Py_END_ALLOW_THREADS

      int *osp = (outside.size() == 0 ? NULL : &outside.front());
      py_outside = c_array_to_python(osp, outside.size());
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  PyObject *py_vo = PyTuple_New(2);
  PyTuple_SetItem(py_vo, 0, py_values);
  PyTuple_SetItem(py_vo, 1, py_outside);

  return py_vo;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *interpolate_volume_gradient(PyObject *, PyObject *args)
{
  PyObject *py_vertices, *py_vtransform, *py_data, *py_gradients = NULL;
  const char *method;
  if (!PyArg_ParseTuple(args, const_cast<char *>("OOOs|O"),
			&py_vertices, &py_vtransform, &py_data,	&method,
			&py_gradients))
    return NULL;

  PyObject *py_outside;
  try
    {
      FArray vertices = array_from_python(py_vertices, 2,
					  Numeric_Array::Float);
      int n = vertices.size(0);
      check_array_size(vertices, n, 3);
      FArray vcontig = vertices.contiguous_array();
      float (*varray)[3] = reinterpret_cast<float(*)[3]>(vcontig.values());

      float vtransform[3][4];
      float *vt = static_cast<float*>(&vtransform[0][0]);
      python_array_to_c(py_vtransform, vt, 3, 4);

      Numeric_Array data = array_from_python(py_data, 3);

      Interpolate::Interpolation_Method m = interpolation_method(method);

      if (py_gradients == NULL)
	py_gradients = python_float_array(n, 3);
      bool allow_data_copy = false;
      FArray gradients = array_from_python(py_gradients, 2,
					   Numeric_Array::Float,
					   allow_data_copy);
      check_array_size(gradients, n, 3, true);

      std::vector<int> outside;

      float (*grad)[3] = reinterpret_cast<float (*)[3]>(gradients.values());

      Py_BEGIN_ALLOW_THREADS
      Interpolate::interpolate_volume_gradient(varray, n, vtransform, data, m,
					       grad, outside);

      Py_END_ALLOW_THREADS

      int *osp = (outside.size() == 0 ? NULL : &outside.front());
      py_outside = c_array_to_python(osp, outside.size());
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  PyObject *py_go = PyTuple_New(2);
  PyTuple_SetItem(py_go, 0, py_gradients);
  PyTuple_SetItem(py_go, 1, py_outside);

  return py_go;
}

// ----------------------------------------------------------------------------
//
static Interpolate::Interpolation_Method interpolation_method(const char *mname)
{
  if (strcmp(mname, "linear") == 0)
    return Interpolate::INTERP_LINEAR;
  else if (strcmp(mname, "nearest") == 0)
    return Interpolate::INTERP_NEAREST;
  else
    {
      std::ostringstream msg;
      msg << "Interpolation method must be 'linear' or 'nearest', got "
	  << mname;
      throw std::runtime_error(msg.str());
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *interpolate_colormap(PyObject *, PyObject *args)
{
  PyObject *py_values, *color_data_values, *rgba_colors;
  PyObject *rgba_above_value_range, *rgba_below_value_range, *py_rgba = NULL;
  if (!PyArg_ParseTuple(args, const_cast<char *>("OOOOO|O"),
			&py_values, &color_data_values, &rgba_colors,
			&rgba_above_value_range, &rgba_below_value_range,
			&py_rgba))
    return NULL;

  try
    {
      FArray values = array_from_python(py_values, 1, Numeric_Array::Float);
      FArray vcontig = values.contiguous_array();

      FArray cdv = array_from_python(color_data_values, 1, Numeric_Array::Float);
      FArray cdvcontig = cdv.contiguous_array();

      FArray cv = array_from_python(rgba_colors, 2, Numeric_Array::Float);
      FArray cvcontig = cv.contiguous_array();
      check_array_size(cv, cdv.size(), 4);
      float (*cva)[4] = reinterpret_cast<float(*)[4]>(cvcontig.values());

      float rgba_above[4], rgba_below[4];
      python_array_to_c(rgba_above_value_range, rgba_above, 4);
      python_array_to_c(rgba_below_value_range, rgba_below, 4);

      int n = values.size();
      if (py_rgba == NULL)
	py_rgba = python_float_array(n, 4);
      bool allow_data_copy = false;
      FArray rgba = array_from_python(py_rgba, 2, Numeric_Array::Float,
				      allow_data_copy);
      check_array_size(rgba, n, 4, true);
      float (*rgbav)[4] = reinterpret_cast<float(*)[4]>(rgba.values());

      Py_BEGIN_ALLOW_THREADS
      Interpolate::interpolate_colormap(vcontig.values(), n,
					cdvcontig.values(), cdv.size(),
					cva, rgba_above, rgba_below, rgbav);
      Py_END_ALLOW_THREADS
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  return py_rgba;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *set_outside_volume_colors(PyObject *, PyObject *args)
{
  PyObject *py_outside, *py_rgba_outside_volume, *py_rgba;
  if (!PyArg_ParseTuple(args, const_cast<char *>("OOO"),
			&py_outside, &py_rgba_outside_volume, &py_rgba))
    return NULL;

  try
    {
      IArray outside = array_from_python(py_outside, 1, Numeric_Array::Int);
      IArray ocontig = outside.contiguous_array();

      float rgba_outside_volume[4];
      python_array_to_c(py_rgba_outside_volume, rgba_outside_volume, 4);

      bool allow_data_copy = false;
      FArray rgba = array_from_python(py_rgba, 2, Numeric_Array::Float,
				      allow_data_copy);
      check_array_size(rgba, rgba.size(0), 4, true);
      float (*rgbav)[4] = reinterpret_cast<float(*)[4]>(rgba.values());

      Py_BEGIN_ALLOW_THREADS
      Interpolate::set_outside_volume_colors(ocontig.values(), ocontig.size(),
					     rgba_outside_volume, rgbav);
      Py_END_ALLOW_THREADS
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static void check_array_size(FArray &a, int n, int m, bool require_contiguous)
{
  if (a.size(0) != n)
    {
      std::ostringstream msg;
      msg << "Array size "
	  << a.size(0)
	  << " does not match other array argument size "
	  << n;
      throw std::runtime_error(msg.str());
    }
  if (a.size(1) != m)
    {
      std::ostringstream msg;
      msg << "The 2nd dimension of array must have size "
	  << m
	  << ", got "
	  << a.size(1);
      throw std::runtime_error(msg.str());
    }
  if (require_contiguous && !a.is_contiguous())
    throw std::runtime_error("Array is non-contiguous");
}

// ----------------------------------------------------------------------------
//
static void check_array_size(FArray &a, int n, bool require_contiguous)
{
  if (a.size(0) != n)
    {
      std::ostringstream msg;
      msg << "Array size "
	  << a.size(0)
	  << " does not match other array argument size "
	  << n;
      throw std::runtime_error(msg.str());
    }
  if (require_contiguous && !a.is_contiguous())
    throw std::runtime_error("Array is non-contiguous");
}
