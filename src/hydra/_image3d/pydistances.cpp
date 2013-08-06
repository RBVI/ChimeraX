// ----------------------------------------------------------------------------
// Python interface to distance routines.
//
#include <Python.h>			// use PyObject

#include <stdexcept>			// use std::runtime_error
#include <sstream>			// use std::ostringstream

#include "distances.h"			// use distances_*()
#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// Numeric_Array

using Reference_Counted_Array::Numeric_Array;

typedef Reference_Counted_Array::Array<float> FArray;

// ----------------------------------------------------------------------------
//
static void check_array_size(FArray &a, int n, int m,
			     bool require_contiguous = false);
static void check_array_size(FArray &a, int n,
			     bool require_contiguous = false);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_distances_from_origin(PyObject *, PyObject *args)
{
  PyObject *py_points, *py_origin, *py_distances;
  if (!PyArg_ParseTuple(args, const_cast<char *>("OOO"),
			&py_points, &py_origin, &py_distances))
    return NULL;

  try
    {
      FArray points = array_from_python(py_points, 2, Numeric_Array::Float);
      check_array_size(points, points.size(0), 3);
      FArray pcontig = points.contiguous_array();
      float (*parray)[3] = reinterpret_cast<float(*)[3]>(pcontig.values());

      float origin[3];
      python_array_to_c(py_origin, origin, 3);

      bool allow_data_copy = false;
      FArray distances = array_from_python(py_distances, 1,
					   Numeric_Array::Float,
					   allow_data_copy);
      check_array_size(distances, points.size(0), true);
      float *darray = distances.values();

      Py_BEGIN_ALLOW_THREADS
      Distances::distances_from_origin(parray, points.size(0), origin, darray);
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
extern "C" PyObject *py_distances_perpendicular_to_axis(PyObject *,
							PyObject *args)
{
  PyObject *py_points, *py_origin, *py_axis, *py_distances;
  if (!PyArg_ParseTuple(args, const_cast<char *>("OOOO"),
			&py_points, &py_origin, &py_axis, &py_distances))
    return NULL;

  try
    {
      FArray points = array_from_python(py_points, 2, Numeric_Array::Float);
      check_array_size(points, points.size(0), 3);
      FArray pcontig = points.contiguous_array();
      float (*parray)[3] = reinterpret_cast<float(*)[3]>(pcontig.values());

      float origin[3], axis[3];
      python_array_to_c(py_origin, origin, 3);
      python_array_to_c(py_axis, axis, 3);

      bool allow_data_copy = false;
      FArray distances = array_from_python(py_distances, 1,
					   Numeric_Array::Float,
					   allow_data_copy);
      check_array_size(distances, points.size(0), true);
      float *darray = distances.values();

      Py_BEGIN_ALLOW_THREADS
      Distances::distances_perpendicular_to_axis(parray, points.size(0),
						 origin, axis, darray);
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
extern "C" PyObject *py_distances_parallel_to_axis(PyObject *, PyObject *args)
{
  PyObject *py_points, *py_origin, *py_axis, *py_distances;
  if (!PyArg_ParseTuple(args, const_cast<char *>("OOOO"),
			&py_points, &py_origin, &py_axis, &py_distances))
    return NULL;

  try
    {
      FArray points = array_from_python(py_points, 2, Numeric_Array::Float);
      check_array_size(points, points.size(0), 3);
      FArray pcontig = points.contiguous_array();
      float (*parray)[3] = reinterpret_cast<float(*)[3]>(pcontig.values());

      float origin[3], axis[3];
      python_array_to_c(py_origin, origin, 3);
      python_array_to_c(py_axis, axis, 3);

      bool allow_data_copy = false;
      FArray distances = array_from_python(py_distances, 1,
					   Numeric_Array::Float,
					   allow_data_copy);
      check_array_size(distances, points.size(0), true);
      float *darray = distances.values();

      Py_BEGIN_ALLOW_THREADS
      Distances::distances_parallel_to_axis(parray, points.size(0),
					    origin, axis, darray);
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
extern "C" PyObject *py_maximum_norm(PyObject *, PyObject *args,
				     PyObject *keywds)
{
  FArray points;
  float tf[3][4];
  const char *kwlist[] = {"points", "transform", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_3x4_array, &tf))
    return NULL;

  try
    {
      check_array_size(points, points.size(0), 3, true);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  float (*p)[3] = reinterpret_cast<float(*)[3]>(points.values());
  float n;
  Py_BEGIN_ALLOW_THREADS
  n = Distances::maximum_norm(p, points.size(0), tf);
  Py_END_ALLOW_THREADS
  PyObject *npy = PyFloat_FromDouble(n);
  return npy;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_correlation_gradient(PyObject *, PyObject *args,
					     PyObject *keywds)
{
  FArray point_weights, values, gradients;
  bool about_mean;
  const char *kwlist[] = {"point_weights", "values", "gradients",
			  "about_mean", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n_array, &point_weights,
				   parse_float_n_array, &values,
				   parse_float_n3_array, &gradients,
				   parse_bool, &about_mean))
    return NULL;

  try
    {
      check_array_size(point_weights, point_weights.size(0), true);
      check_array_size(values, point_weights.size(0), true);
      check_array_size(gradients, point_weights.size(0), 3, true);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  float grad[3];
  float (*g)[3] = reinterpret_cast<float(*)[3]>(gradients.values());
  Py_BEGIN_ALLOW_THREADS
  Distances::correlation_gradient(point_weights.values(), point_weights.size(0),
				  values.values(), g, about_mean, &grad[0]);
  Py_END_ALLOW_THREADS
  PyObject *gpy = c_array_to_python(grad, 3);
  return gpy;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_torque(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points, point_weights, forces;
  PyObject *pyw;
  float center[3], *w;
  const char *kwlist[] = {"points", "point_weights", "forces", "center", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&OO&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &points,
				   &pyw,
				   parse_float_n3_array, &forces,
				   parse_float_3_array, &center))
    return NULL;

  if (pyw == Py_None)
    w = NULL;
  else if (parse_float_n_array(pyw, &point_weights))
    w = point_weights.values();
  else
    return NULL;

  try
    {
      check_array_size(points, points.size(0), 3, true);
      if (w)
	check_array_size(point_weights, points.size(0), true);
      check_array_size(forces, points.size(0), 3, true);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  float tor[3];
  float (*p)[3] = reinterpret_cast<float(*)[3]>(points.values());
  float (*f)[3] = reinterpret_cast<float(*)[3]>(forces.values());
  Py_BEGIN_ALLOW_THREADS
  Distances::torque(p, points.size(0), w, f, center, &tor[0]);
  Py_END_ALLOW_THREADS
  PyObject *torpy = c_array_to_python(tor, 3);
  return torpy;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_torques(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points, forces, torques;
  float center[3];
  const char *kwlist[] = {"points", "center", "forces", "torques", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_3_array, &center,
				   parse_float_n3_array, &forces,
				   parse_writable_float_n3_array, &torques))
    return NULL;

  try
    {
      check_array_size(points, points.size(0), 3, true);
      check_array_size(forces, points.size(0), 3, true);
      check_array_size(torques, points.size(0), 3, true);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  float (*p)[3] = reinterpret_cast<float(*)[3]>(points.values());
  float (*f)[3] = reinterpret_cast<float(*)[3]>(forces.values());
  float (*t)[3] = reinterpret_cast<float(*)[3]>(torques.values());
  Py_BEGIN_ALLOW_THREADS
  Distances::torques(p, points.size(0), center, f, t);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_correlation_torque(PyObject *, PyObject *args,
					   PyObject *keywds)
{
  FArray points, point_weights, values, gradients;
  float center[3];
  bool about_mean;
  const char *kwlist[] = {"points", "point_weights", "values", "gradients",
			  "center", "about_mean", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_n_array, &point_weights,
				   parse_float_n_array, &values,
				   parse_float_n3_array, &gradients,
				   parse_float_3_array, &center,
				   parse_bool, &about_mean))
    return NULL;

  try
    {
      check_array_size(points, points.size(0), 3, true);
      check_array_size(point_weights, points.size(0), true);
      check_array_size(values, points.size(0), true);
      check_array_size(gradients, points.size(0), 3, true);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  float tor[3];
  float (*p)[3] = reinterpret_cast<float(*)[3]>(points.values());
  float (*g)[3] = reinterpret_cast<float(*)[3]>(gradients.values());
  Py_BEGIN_ALLOW_THREADS
  Distances::correlation_torque(p, points.size(0), point_weights.values(),
				values.values(), g, center, about_mean,
				&tor[0]);
  Py_END_ALLOW_THREADS
  PyObject *torpy = c_array_to_python(tor, 3);
  return torpy;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_correlation_torque2(PyObject *, PyObject *args,
					    PyObject *keywds)
{
  FArray point_weights, values, torques;
  bool about_mean;
  const char *kwlist[] = {"point_weights", "values", "torques",
			  "about_mean", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n_array, &point_weights,
				   parse_float_n_array, &values,
				   parse_float_n3_array, &torques,
				   parse_bool, &about_mean))
    return NULL;

  try
    {
      check_array_size(point_weights, point_weights.size(0), true);
      check_array_size(values, point_weights.size(0), true);
      check_array_size(torques, point_weights.size(0), 3, true);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  float tor[3];
  float (*t)[3] = reinterpret_cast<float(*)[3]>(torques.values());
  Py_BEGIN_ALLOW_THREADS
  Distances::correlation_torque2(point_weights.values(), point_weights.size(0),
				 values.values(), t, about_mean, &tor[0]);
  Py_END_ALLOW_THREADS

  PyObject *torpy = c_array_to_python(tor, 3);
  return torpy;
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
