// ----------------------------------------------------------------------------
// Python interface to distance routines.
//
#include <Python.h>			// use PyObject

#include "distances.h"			// use distances_*()
#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// Numeric_Array

using Reference_Counted_Array::Numeric_Array;

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_distances_from_origin(PyObject *, PyObject *args)
{
  FArray points, distances;
  float origin[3];
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&O&"),
			parse_float_n3_array, &points,
			parse_float_3_array, &origin[0],
			parse_writable_float_n_array, &distances) ||
      !check_array_size(distances, points.size(0), true))
    return NULL;

  FArray pcontig = points.contiguous_array();
  float (*parray)[3] = reinterpret_cast<float(*)[3]>(pcontig.values());
  float *darray = distances.values();

  Py_BEGIN_ALLOW_THREADS
    Distances::distances_from_origin(parray, points.size(0), origin, darray);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_distances_perpendicular_to_axis(PyObject *,
							PyObject *args)
{
  FArray points, distances;
  float origin[3], axis[3];
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&O&O&"),
			parse_float_n3_array, &points,
			parse_float_3_array, &origin[0],
			parse_float_3_array, &axis[0],
			parse_writable_float_n_array, &distances) ||
      !check_array_size(distances, points.size(0), true))
    return NULL;

  FArray pcontig = points.contiguous_array();
  float (*parray)[3] = reinterpret_cast<float(*)[3]>(pcontig.values());
  float *darray = distances.values();

  Py_BEGIN_ALLOW_THREADS
    Distances::distances_perpendicular_to_axis(parray, points.size(0),
					       origin, axis, darray);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_distances_parallel_to_axis(PyObject *, PyObject *args)
{
  FArray points, distances;
  float origin[3], axis[3];
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&O&O&O&"),
			parse_float_n3_array, &points,
			parse_float_3_array, &origin[0],
			parse_float_3_array, &axis[0],
			parse_writable_float_n_array, &distances) ||
      !check_array_size(distances, points.size(0), true))
    return NULL;

  FArray pcontig = points.contiguous_array();
  float (*parray)[3] = reinterpret_cast<float(*)[3]>(pcontig.values());
  float *darray = distances.values();

  Py_BEGIN_ALLOW_THREADS
    Distances::distances_parallel_to_axis(parray, points.size(0),
					  origin, axis, darray);
  Py_END_ALLOW_THREADS

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
				   parse_float_3x4_array, &tf) ||
      !check_array_size(points, points.size(0), 3, true))
    return NULL;

  float (*p)[3] = reinterpret_cast<float(*)[3]>(points.values());
  float n;
  Py_BEGIN_ALLOW_THREADS
  n = Distances::maximum_norm(p, points.size(0), tf);
  Py_END_ALLOW_THREADS
  PyObject *npy = PyFloat_FromDouble(n);
  return npy;
}
