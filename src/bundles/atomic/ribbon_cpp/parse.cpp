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
#include <map>				// use std::map

#include <Python.h>			// use PyObject

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>		// use PyArray_*(), NPY_*

#include "parse.h"			// use Residues

// -----------------------------------------------------------------------------
//
extern "C" int parse_residues(PyObject *arg, void *res)
{
  import_array(); // Initialize numpy.
    
  if (!PyArray_Check(arg))
    {
      PyErr_SetString(PyExc_TypeError, "residues argument is not a numpy array");
      return 0;
    }

  PyArrayObject *a = static_cast<PyArrayObject *>(static_cast<void *>(arg));
  if (PyArray_TYPE(a) != NPY_UINTP)
    {
      PyErr_SetString(PyExc_TypeError, "residues argument numpy array is not of type uintp");
      return 0;
    }

  if (PyArray_NDIM(a) != 1)
    {
      PyErr_SetString(PyExc_TypeError, "residues argument numpy array is not 1 dimensional");
      return 0;
    }

  // Check if array is contiguous.
  if (PyArray_STRIDE(a,0) != static_cast<int>(sizeof(void *)))
    {
      PyErr_SetString(PyExc_TypeError, "residues argument numpy array is not contiguous");
      return 0;
    }

  Residues *r = static_cast<Residues *>(res);
  r->count = PyArray_DIM(a,0);
  r->pointers = static_cast<Residue **>(PyArray_DATA(a));

  return 1;
}

// -----------------------------------------------------------------------------
//
extern "C" int parse_string_float_map(PyObject *arg, void *sf)
{
  if (!PyDict_Check(arg))
    {
      PyErr_SetString(PyExc_TypeError, "argument is not a dictionary");
      return 0;
    }

  std::map<std::string, float> *sfmap = static_cast<std::map<std::string, float> *>(sf);
  Py_ssize_t index = 0;
  PyObject *key;
  PyObject *value;
  while (PyDict_Next(arg, &index, &key, &value))
    {
      if (!PyUnicode_Check(key))
	{
	  PyErr_SetString(PyExc_TypeError, "dictionary argument key is not a string");
	  return 0;
	}
      if (!PyFloat_Check(value))
	{
	  PyErr_SetString(PyExc_TypeError, "dictionary argument value is not a float");
	  return 0;
	}
      (*sfmap)[PyUnicode_AsUTF8AndSize(key,NULL)] = PyFloat_AsDouble(value);
    }
  return 1;
}
