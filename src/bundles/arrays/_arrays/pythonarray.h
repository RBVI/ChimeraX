// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
// Convert multi-dimensional arrays between Python and C++.
//
#ifndef PYTHONARRAY_HEADER_INCLUDED
#define PYTHONARRAY_HEADER_INCLUDED

#include <Python.h>
#include <cstdint>	// use std::int64_t
#include <vector>       // use std::vector<>

#include "imex.h"

#include "rcarray.h"	// use Numeric_Array

using std::int64_t;

using Reference_Counted_Array::Numeric_Array;

//
// Return false if python object is not an array of specified dimension.
//
ARRAYS_IMEX bool array_from_python(PyObject *array, int dim, Numeric_Array *na,
                                   bool allow_data_copy = true);

//
// Return false if python object is not an array of specified
// dimension or does not have the specified value type.
//
ARRAYS_IMEX bool array_from_python(PyObject *array, int dim,
                                   Numeric_Array::Value_Type required_type,
                                   Numeric_Array *na,
                                   bool allow_data_copy = true);

//
// Recover numpy Python array used to create a C++ array.
// Returns NULL if there is no Python array.
//
ARRAYS_IMEX PyObject *array_python_source(const Reference_Counted_Array::Untyped_Array &a,
                                          bool incref = true);

//
// Routines for parsing array arguments with PyArg_ParseTuple().
//
extern "C" {

// float [] value
//
// Example usage:
//
// float origin[3];
// const char *kwlist[] = {"origin", NULL};
// if (!PyArg_ParseTupleAndKeywords(args, keywds,
//                                  const_cast<char *>("O&"), (char **)kwlist,
//			            parse_float_3_array, &origin[0])
//     return NULL;
//  
ARRAYS_IMEX int parse_float_3_array(PyObject *arg, void *f3);
ARRAYS_IMEX int parse_float_4_array(PyObject *arg, void *f4);
ARRAYS_IMEX int parse_float_3x3_array(PyObject *arg, void *f3x3);
ARRAYS_IMEX int parse_float_3x4_array(PyObject *arg, void *f3x4);

// FArray value
//
// Example usage:
//
// FArray vertices;
// const char *kwlist[] = {"vertices", NULL};
// if (!PyArg_ParseTupleAndKeywords(args, keywds,
//                                  const_cast<char *>("O&"), (char **)kwlist,
//			            parse_float_n3_array, &vertices)
//     return NULL;
//  
ARRAYS_IMEX int parse_float_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_float_n_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_float_n2_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_float_n3_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_float_n4_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_float_2d_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_contiguous_float_4x4_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_contiguous_float_n44_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_writable_float_n_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_writable_float_n3_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_writable_float_n4_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_writable_float_n9_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_writable_float_2d_array(PyObject *arg, void *farray);
ARRAYS_IMEX int parse_writable_float_3d_array(PyObject *arg, void *farray);

// double [] value
ARRAYS_IMEX int parse_double_3_array(PyObject *arg, void *d3);
ARRAYS_IMEX int parse_double_3x3_array(PyObject *arg, void *d3x3);
ARRAYS_IMEX int parse_double_3x4_array(PyObject *arg, void *d3x4);

// DArray value
ARRAYS_IMEX int parse_double_n_array(PyObject *arg, void *darray);
ARRAYS_IMEX int parse_double_n3_array(PyObject *arg, void *darray);
ARRAYS_IMEX int parse_contiguous_double_3x4_array(PyObject *arg, void *darray);
ARRAYS_IMEX int parse_contiguous_double_n34_array(PyObject *arg, void *darray);
ARRAYS_IMEX int parse_writable_double_n_array(PyObject *arg, void *darray);
ARRAYS_IMEX int parse_writable_double_n3_array(PyObject *arg, void *darray);

// int [] value
ARRAYS_IMEX int parse_int_3_array(PyObject *arg, void *i3);

// IArray value
//
// IArray vertices;
// const char *kwlist[] = {"triangles", NULL};
// if (!PyArg_ParseTupleAndKeywords(args, keywds,
//                                  const_cast<char *>("O&"), (char **)kwlist,
//			            parse_int_n3_array, &triangles)
//     return NULL;
//
ARRAYS_IMEX int parse_int_n_array(PyObject *arg, void *iarray);
ARRAYS_IMEX int parse_int_n2_array(PyObject *arg, void *iarray);
ARRAYS_IMEX int parse_int_n3_array(PyObject *arg, void *iarray);
ARRAYS_IMEX int parse_int_2d_array(PyObject *arg, void *iarray);
ARRAYS_IMEX int parse_writable_int_n_array(PyObject *arg, void *iarray);
ARRAYS_IMEX int parse_writable_int_n3_array(PyObject *arg, void *iarray);
ARRAYS_IMEX int parse_writable_int_2d_array(PyObject *arg, void *iarray);

// BArray value (unsigned char)
ARRAYS_IMEX int parse_uint8_n_array(PyObject *arg, void *barray);
ARRAYS_IMEX int parse_uint8_n2_array(PyObject *arg, void *barray);
ARRAYS_IMEX int parse_uint8_n3_array(PyObject *arg, void *barray);
ARRAYS_IMEX int parse_uint8_n4_array(PyObject *arg, void *barray);
ARRAYS_IMEX int parse_writable_uint8_n_array(PyObject *arg, void *barray);

// Numeric_Array value, any numeric type
ARRAYS_IMEX int parse_array(PyObject *arg, void *array);
ARRAYS_IMEX int parse_1d_array(PyObject *arg, void *array);
ARRAYS_IMEX int parse_2d_array(PyObject *arg, void *array);
ARRAYS_IMEX int parse_3d_array(PyObject *arg, void *array);
ARRAYS_IMEX int parse_writable_array(PyObject *arg, void *array);
ARRAYS_IMEX int parse_writable_2d_array(PyObject *arg, void *array);
ARRAYS_IMEX int parse_writable_3d_array(PyObject *arg, void *array);
ARRAYS_IMEX int parse_writable_4d_array(PyObject *arg, void *array);

// single bool value
ARRAYS_IMEX int parse_bool(PyObject *arg, void *b);

// Array<char> value
ARRAYS_IMEX int parse_string_array(PyObject *arg, void *carray);

// single void * value from Python long
ARRAYS_IMEX int parse_voidp(PyObject *arg, void **p);
}

ARRAYS_IMEX bool check_array_size(FArray &a, int64_t n, int64_t m, bool require_contiguous = false);
ARRAYS_IMEX bool check_array_size(FArray &a, int64_t n, bool require_contiguous = false);

//
// Convert a one dimensional sequences of known length from Python to C.
// python_array_to_c() returns false if python object is not
// array of correct size.
//
ARRAYS_IMEX bool python_array_to_c(PyObject *a, int *values, int64_t size);
ARRAYS_IMEX bool python_array_to_c(PyObject *a, float *values, int64_t size);
ARRAYS_IMEX bool python_array_to_c(PyObject *a, float *values, int64_t size0, int64_t size1);
ARRAYS_IMEX bool python_array_to_c(PyObject *a, double *values, int64_t size);
ARRAYS_IMEX bool python_array_to_c(PyObject *a, double *values, int64_t size0, int64_t size1);

ARRAYS_IMEX bool float_2d_array_values(PyObject *farray, int64_t n2, float **f, int64_t *size);

//
// Convert C arrays to Python Numpy arrays.
//
ARRAYS_IMEX PyObject *c_array_to_python(const int *values, int64_t size);
ARRAYS_IMEX PyObject *c_array_to_python(const std::vector<int> &values);
ARRAYS_IMEX PyObject *c_array_to_python(const std::vector<int64_t> &values);
ARRAYS_IMEX PyObject *c_array_to_python(const std::vector<int> &values, int64_t size0, int64_t size1);
ARRAYS_IMEX PyObject *c_array_to_python(const std::vector<float> &values);
ARRAYS_IMEX PyObject *c_array_to_python(const std::vector<float> &values, int64_t size0, int64_t size1);
ARRAYS_IMEX PyObject *c_array_to_python(const float *values, int64_t size);
ARRAYS_IMEX PyObject *c_array_to_python(const double *values, int64_t size);
ARRAYS_IMEX PyObject *c_array_to_python(const int *values, int64_t size0, int64_t size1);
ARRAYS_IMEX PyObject *c_array_to_python(const float *values, int64_t size0, int64_t size1);
ARRAYS_IMEX PyObject *c_array_to_python(const double *values, int64_t size0, int64_t size1);

//
// Create an uninitialized Numpy array.
//
ARRAYS_IMEX PyObject *python_bool_array(int64_t size, unsigned char **data = NULL);
ARRAYS_IMEX PyObject *python_uint8_array(int64_t size, unsigned char **data = NULL);
ARRAYS_IMEX PyObject *python_uint8_array(int64_t size1, int64_t size2, unsigned char **data = NULL);
ARRAYS_IMEX PyObject *python_int_array(int64_t size, int **data = NULL);
ARRAYS_IMEX PyObject *python_int_array(int64_t size1, int64_t size2, int **data = NULL);
ARRAYS_IMEX PyObject *python_unsigned_int_array(int64_t size1, int64_t size2, int64_t size3,
						unsigned int **data = NULL);
ARRAYS_IMEX PyObject *python_float_array(int64_t size, float **data = NULL);
ARRAYS_IMEX PyObject *python_float_array(int64_t size1, int64_t size2, float **data = NULL);
ARRAYS_IMEX PyObject *python_float_array(int64_t size1, int64_t size2, int64_t size3,
					 float **data = NULL);
ARRAYS_IMEX PyObject *python_double_array(int64_t size, double **data = NULL);
ARRAYS_IMEX PyObject *python_double_array(int64_t size1, int64_t size2, double **data = NULL);
ARRAYS_IMEX PyObject *python_double_array(int64_t size1, int64_t size2, int64_t size3,
					  double **data = NULL);
ARRAYS_IMEX PyObject *python_voidp_array(int64_t size, void ***data = NULL);
ARRAYS_IMEX PyObject *python_object_array(int64_t size, PyObject **data = NULL);

ARRAYS_IMEX PyObject *resized_2d_array(PyObject *array, int64_t size0, int64_t size1);

ARRAYS_IMEX PyObject *python_none();
ARRAYS_IMEX PyObject *python_bool(bool b);
ARRAYS_IMEX PyObject *python_voidp(void *p);
  
ARRAYS_IMEX PyObject *python_tuple(PyObject *o1, PyObject *o2);
ARRAYS_IMEX PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3);
ARRAYS_IMEX PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3, PyObject *a4);
ARRAYS_IMEX PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3, PyObject *a4, PyObject *a5);

#endif
