// ----------------------------------------------------------------------------
// Convert multi-dimensional arrays between Python and C++.
//
#ifndef PYTHONARRAY_HEADER_INCLUDED
#define PYTHONARRAY_HEADER_INCLUDED

#include <Python.h>
#include <vector>		// use std::vector<>

#include "rcarray.h"		// use Numeric_Array

using Reference_Counted_Array::Numeric_Array;
using Reference_Counted_Array::Untyped_Array;

//
// Return false if python object is not an array of specified dimension.
//
bool array_from_python(PyObject *array, int dim, Numeric_Array *na,
		       bool allow_data_copy = true);

//
// Return false if python object is not an array of specified
// dimension or does not have the specified value type.
//
bool array_from_python(PyObject *array, int dim,
		       Numeric_Array::Value_Type required_type,
		       Numeric_Array *na,
		       bool allow_data_copy = true);

//
// Recover numpy Python array used to create a C++ array.
// Returns NULL if there is no Python array.
//
PyObject *array_python_source(const Untyped_Array &a);

//
// Routines for parsing array arguments with PyArg_ParseTuple().
//
extern "C" {
int parse_bool(PyObject *arg, void *b);
int parse_float_n3_array(PyObject *arg, void *farray);
int parse_writable_float_n3_array(PyObject *arg, void *farray);
int parse_double_n3_array(PyObject *arg, void *darray);
int parse_writable_double_n3_array(PyObject *arg, void *darray);
int parse_uint8_n_array(PyObject *arg, void *carray);
int parse_uint8_n4_array(PyObject *arg, void *carray);
int parse_float_n4_array(PyObject *arg, void *farray);
int parse_writable_float_n4_array(PyObject *arg, void *farray);
int parse_float_n_array(PyObject *arg, void *farray);
int parse_writable_float_n_array(PyObject *arg, void *farray);
int parse_double_n_array(PyObject *arg, void *farray);
int parse_writable_double_n_array(PyObject *arg, void *farray);
int parse_int_3_array(PyObject *arg, void *i3);
int parse_float_3_array(PyObject *arg, void *f3);
int parse_double_3_array(PyObject *arg, void *f3);
int parse_float_4_array(PyObject *arg, void *f4);
int parse_float_3x4_array(PyObject *arg, void *f3x4);
int parse_double_3x4_array(PyObject *arg, void *d3x4);
int parse_writable_float_3d_array(PyObject *arg, void *farray);
int parse_int_n_array(PyObject *arg, void *iarray);
int parse_int_n2_array(PyObject *arg, void *iarray);
int parse_int_n3_array(PyObject *arg, void *iarray);
int parse_writable_int_n_array(PyObject *arg, void *iarray);
int parse_writable_int_n3_array(PyObject *arg, void *iarray);
int parse_1d_array(PyObject *arg, void *array);
int parse_2d_array(PyObject *arg, void *array);
int parse_3d_array(PyObject *arg, void *array);
int parse_array(PyObject *arg, void *array);
int parse_writable_array(PyObject *arg, void *array);
int parse_float_array(PyObject *arg, void *array);
int parse_writable_3d_array(PyObject *arg, void *array);
int parse_writable_4d_array(PyObject *arg, void *array);
int parse_string_array(PyObject *arg, void *carray);
}

bool check_array_size(FArray &a, int n, int m, bool require_contiguous = false);
bool check_array_size(FArray &a, int n, bool require_contiguous = false);

//
// Convert a one dimensional sequences of known length from Python to C.
// python_array_to_c() returns false if python object is not
// array of correct size.
//
bool python_array_to_c(PyObject *a, int *values, int size);
bool python_array_to_c(PyObject *a, float *values, int size);
bool python_array_to_c(PyObject *a, float *values, int size0, int size1);
bool python_array_to_c(PyObject *a, double *values, int size);
bool python_array_to_c(PyObject *a, double *values, int size0, int size1);

bool float_2d_array_values(PyObject *farray, int n2, float **f, int *size);

//
// Convert C arrays to Python Numeric arrays.
//
PyObject *c_array_to_python(const int *values, int size);
PyObject *c_array_to_python(const std::vector<int> &i);
PyObject *c_array_to_python(const float *values, int size);
PyObject *c_array_to_python(const double *values, int size);
PyObject *c_array_to_python(const int *values, int size0, int size1);
PyObject *c_array_to_python(const float *values, int size0, int size1);
PyObject *c_array_to_python(const double *values, int size0, int size1);

//
// Create an uninitialized Numeric array.
//
PyObject *python_uint8_array(int size, unsigned char **data = NULL);
PyObject *python_uint8_array(int size1, int size2, unsigned char **data = NULL);
PyObject *python_char_array(int size1, int size2, char **data = NULL);
PyObject *python_string_array(int size, int string_length, char **data = NULL);
PyObject *python_int_array(int size, int **data = NULL);
PyObject *python_int_array(int size1, int size2, int **data = NULL);
PyObject *python_unsigned_int_array(int size1, int size2, int size3, unsigned int **data = NULL);
PyObject *python_float_array(int size, float **data = NULL);
PyObject *python_float_array(int size1, int size2, float **data = NULL);
PyObject *python_float_array(int size1, int size2, int size3, float **data = NULL);
PyObject *python_double_array(int size, double **data = NULL);

PyObject *python_tuple(PyObject *o1, PyObject *o2);
PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3);
PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3, PyObject *a4);

#endif
