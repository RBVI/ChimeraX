// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
// Python interface to distance routines.
//
#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// Numeric_Array
#include "distances.h"			// use distances_*()
#include "distancespy.h"

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

  return python_none();
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

  return python_none();
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

  return python_none();
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
