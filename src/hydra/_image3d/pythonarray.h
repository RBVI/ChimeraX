// ----------------------------------------------------------------------------
// Convert multi-dimensional arrays between Python and C++.
//
#ifndef PYTHONARRAY_HEADER_INCLUDED
#define PYTHONARRAY_HEADER_INCLUDED

#include <Python.h>
#include <vector>		// use std::vector<>

#include "rcarray.h"		// use Numeric_Array
#include "volumearray_config.h"	// use VOLUMEARRAY_IMEX

using Reference_Counted_Array::Numeric_Array;
using Reference_Counted_Array::Untyped_Array;

//
// Throw std::runtime_error() if python object is not an array of specified
// dimension.
//
VOLUMEARRAY_IMEX
Numeric_Array array_from_python(PyObject *array, int dim,
				bool allow_data_copy = true);

//
// Throw std::runtime_error() if python object is not an array of specified
// dimension or does not have the specified value type.
//
VOLUMEARRAY_IMEX
Numeric_Array array_from_python(PyObject *array, int dim,
				Numeric_Array::Value_Type required_type,
				bool allow_data_copy = true);

//
// Recover numpy Python array used to create a C++ array.
// Returns NULL if there is no Python array.
//
VOLUMEARRAY_IMEX
PyObject *array_python_source(const Untyped_Array &a);

//
// Routines for parsing array arguments with PyArg_ParseTuple().
//
extern "C" {
VOLUMEARRAY_IMEX int parse_bool(PyObject *arg, void *b);
VOLUMEARRAY_IMEX int parse_float_n3_array(PyObject *arg, void *farray);
VOLUMEARRAY_IMEX int parse_writable_float_n3_array(PyObject *arg, void *farray);
VOLUMEARRAY_IMEX int parse_float_n_array(PyObject *arg, void *farray);
VOLUMEARRAY_IMEX int parse_int_3_array(PyObject *arg, void *i3);
VOLUMEARRAY_IMEX int parse_float_3_array(PyObject *arg, void *f3);
VOLUMEARRAY_IMEX int parse_double_3_array(PyObject *arg, void *f3);
VOLUMEARRAY_IMEX int parse_float_3x4_array(PyObject *arg, void *f3x4);
VOLUMEARRAY_IMEX int parse_writable_float_3d_array(PyObject *arg, void *farray);
VOLUMEARRAY_IMEX int parse_int_n_array(PyObject *arg, void *iarray);
VOLUMEARRAY_IMEX int parse_int_n2_array(PyObject *arg, void *iarray);
VOLUMEARRAY_IMEX int parse_int_n3_array(PyObject *arg, void *iarray);
VOLUMEARRAY_IMEX int parse_writable_int_n_array(PyObject *arg, void *iarray);
VOLUMEARRAY_IMEX int parse_writable_int_n3_array(PyObject *arg, void *iarray);
VOLUMEARRAY_IMEX int parse_3d_array(PyObject *arg, void *array);
VOLUMEARRAY_IMEX int parse_writable_3d_array(PyObject *arg, void *array);
VOLUMEARRAY_IMEX int parse_string_array(PyObject *arg, void *carray);
}

//
// Convert a one dimensional sequences of known length from Python to C.
// python_array_to_c() throws std::runtime_error() if python object is not
// array of correct size.
//
VOLUMEARRAY_IMEX
void python_array_to_c(PyObject *a, int *values, int size);
VOLUMEARRAY_IMEX
void python_array_to_c(PyObject *a, float *values, int size);
VOLUMEARRAY_IMEX
void python_array_to_c(PyObject *a, float *values, int size0, int size1);
VOLUMEARRAY_IMEX
void python_array_to_c(PyObject *a, double *values, int size);
VOLUMEARRAY_IMEX
void python_array_to_c(PyObject *a, double *values, int size0, int size1);

VOLUMEARRAY_IMEX
bool float_2d_array_values(PyObject *farray, int n2, float **f, int *size);

//
// Convert C arrays to Python Numeric arrays.
//
VOLUMEARRAY_IMEX
PyObject *c_array_to_python(const int *values, int size);
VOLUMEARRAY_IMEX
PyObject *c_array_to_python(const std::vector<int> &i);
VOLUMEARRAY_IMEX
PyObject *c_array_to_python(const float *values, int size);
VOLUMEARRAY_IMEX
PyObject *c_array_to_python(const double *values, int size);
VOLUMEARRAY_IMEX
PyObject *c_array_to_python(const int *values, int size0, int size1);
VOLUMEARRAY_IMEX
PyObject *c_array_to_python(const float *values, int size0, int size1);
VOLUMEARRAY_IMEX
PyObject *c_array_to_python(const double *values, int size0, int size1);

//
// Create an uninitialized Numeric array.
//
VOLUMEARRAY_IMEX
PyObject *python_uint8_array(int size, unsigned char **data = NULL);
VOLUMEARRAY_IMEX
PyObject *python_char_array(int size1, int size2, char **data = NULL);
VOLUMEARRAY_IMEX
PyObject *python_string_array(int size, int string_length, char **data = NULL);
VOLUMEARRAY_IMEX
PyObject *python_int_array(int size, int **data = NULL);
VOLUMEARRAY_IMEX
PyObject *python_int_array(int size1, int size2, int **data = NULL);
VOLUMEARRAY_IMEX
PyObject *python_unsigned_int_array(int size1, int size2, int size3, unsigned int **data = NULL);
VOLUMEARRAY_IMEX
PyObject *python_float_array(int size, float **data = NULL);
VOLUMEARRAY_IMEX
PyObject *python_float_array(int size1, int size2, float **data = NULL);
VOLUMEARRAY_IMEX
PyObject *python_float_array(int size1, int size2, int size3, float **data = NULL);

VOLUMEARRAY_IMEX
PyObject *python_tuple(PyObject *o1, PyObject *o2);
VOLUMEARRAY_IMEX
PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3);
VOLUMEARRAY_IMEX
PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3, PyObject *a4);

#endif
