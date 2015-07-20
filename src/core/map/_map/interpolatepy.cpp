// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Python interface to interpolation routines.
//
#include <Python.h>			// use PyObject

#include <vector>			// use std::vector

#include "interpolate.h"		// use interpolate_volume_data()
#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// Numeric_Array

using Reference_Counted_Array::Numeric_Array;

// ----------------------------------------------------------------------------
//
static int parse_interpolation_method(PyObject *arg, void *m);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *interpolate_volume_data(PyObject *, PyObject *args)
{
  FArray vertices, values;
  float vtransform[3][4];
  Numeric_Array data;
  Interpolate::Interpolation_Method method;
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&O&O&|O&"),
			parse_float_n3_array, &vertices,
			parse_float_3x4_array, &(vtransform[0][0]),
			parse_3d_array, &data,
			parse_interpolation_method, &method,
			parse_writable_float_n_array, &values) ||
      (values.dimension() == 1 && !check_array_size(values, vertices.size(0), true)))
    return NULL;

  FArray vcontig = vertices.contiguous_array();
  float (*varray)[3] = reinterpret_cast<float(*)[3]>(vcontig.values());

  int n = vertices.size(0);
  if (values.dimension() == 0)
    parse_writable_float_n_array(python_float_array(n), &values);

  std::vector<int> outside;

  Py_BEGIN_ALLOW_THREADS
    Interpolate::interpolate_volume_data(varray, n, vtransform, data, method,
					 values.values(), outside);
  Py_END_ALLOW_THREADS

  int *osp = (outside.size() == 0 ? NULL : &outside.front());
  PyObject *py_outside = c_array_to_python(osp, outside.size());

  PyObject *py_values = array_python_source(values);
  PyObject *result =  python_tuple(py_values, py_outside);
  return result;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *interpolate_volume_gradient(PyObject *, PyObject *args)
{
  FArray vertices, gradients;
  float vtransform[3][4];
  Numeric_Array data;
  Interpolate::Interpolation_Method method;
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&O&O&|O&"),
			parse_float_n3_array, &vertices,
			parse_float_3x4_array, &(vtransform[0][0]),
			parse_3d_array, &data,
			parse_interpolation_method, &method,
			parse_writable_float_n3_array, &gradients) ||
      (gradients.dimension() == 2 && !check_array_size(gradients, vertices.size(0), 3, true)))
    return NULL;

  FArray vcontig = vertices.contiguous_array();
  float (*varray)[3] = reinterpret_cast<float(*)[3]>(vcontig.values());

  int n = vertices.size(0);
  if (gradients.dimension() == 0)
    parse_writable_float_n3_array(python_float_array(n,3), &gradients);
  float (*grad)[3] = reinterpret_cast<float (*)[3]>(gradients.values());

  std::vector<int> outside;

  Py_BEGIN_ALLOW_THREADS
    Interpolate::interpolate_volume_gradient(varray, n, vtransform, data, method,
					     grad, outside);

  Py_END_ALLOW_THREADS

  int *osp = (outside.size() == 0 ? NULL : &outside.front());
  PyObject *py_outside = c_array_to_python(osp, outside.size());

  PyObject *py_gradients = array_python_source(gradients);
  PyObject *result = python_tuple(py_gradients, py_outside);
  return result;
}

// ----------------------------------------------------------------------------
//
static int parse_interpolation_method(PyObject *arg, void *m)
{
  const char *mname = PyUnicode_AsUTF8(arg);
  if (mname == NULL)
    return 0;

  Interpolate::Interpolation_Method method;
  if (strcmp(mname, "linear") == 0)
    method = Interpolate::INTERP_LINEAR;
  else if (strcmp(mname, "nearest") == 0)
    method = Interpolate::INTERP_NEAREST;
  else
    {
      PyErr_Format(PyExc_TypeError, "Interpolation method must be 'linear' or 'nearest', got %s", mname);
      return 0;
    }
  *static_cast<Interpolate::Interpolation_Method *>(m) = method;
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *interpolate_colormap(PyObject *, PyObject *args)
{
  FArray values, color_data_values, colors, rgba;
  float rgba_above[4], rgba_below[4];
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&O&O&O&|O&"),
			parse_float_n_array, &values,
			parse_float_n_array, &color_data_values,
			parse_float_n4_array, &colors,
			parse_float_4_array, &rgba_above,
			parse_float_4_array, &rgba_below,
			parse_writable_float_n4_array, &rgba) ||
      !check_array_size(colors, color_data_values.size(), 4) ||
      (rgba.dimension() > 0 && !check_array_size(rgba, values.size(0), 4, true)))
    return NULL;

  FArray vcontig = values.contiguous_array();
  FArray cdvcontig = color_data_values.contiguous_array();
  FArray cvcontig = colors.contiguous_array();
  float (*cva)[4] = reinterpret_cast<float(*)[4]>(cvcontig.values());

  int n = values.size();
  if (rgba.dimension() == 0)
    parse_writable_float_n4_array(python_float_array(n, 4), &rgba);
  float (*rgbav)[4] = reinterpret_cast<float(*)[4]>(rgba.values());

  Py_BEGIN_ALLOW_THREADS
    Interpolate::interpolate_colormap(vcontig.values(), n,
				      cdvcontig.values(), cdvcontig.size(),
				      cva, rgba_above, rgba_below, rgbav);
  Py_END_ALLOW_THREADS

  return array_python_source(rgba);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *set_outside_volume_colors(PyObject *, PyObject *args)
{
  IArray outside;
  float rgba_outside_volume[4];
  FArray rgba;
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&O&"),
			parse_int_n_array, &outside,
			parse_float_4_array, &rgba_outside_volume,
			parse_writable_float_n4_array, &rgba))
    return NULL;

  IArray ocontig = outside.contiguous_array();
  float (*rgbav)[4] = reinterpret_cast<float(*)[4]>(rgba.values());

  Py_BEGIN_ALLOW_THREADS
    Interpolate::set_outside_volume_colors(ocontig.values(), ocontig.size(),
					   rgba_outside_volume, rgbav);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}
