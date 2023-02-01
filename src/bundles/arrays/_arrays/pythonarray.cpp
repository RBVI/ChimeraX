// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
//
// #include <iostream>          // use std::cerr for debugging
#include <sstream>		// use std::ostringstream
#include <Python.h>		// use Py_DECREF()

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>      // use PyArray_*(), NPY_*

#define ARRAYS_EXPORT
#include "pythonarray.h"
#include "rcarray.h"            // use Numeric_Array, Release_Data

using Reference_Counted_Array::Numeric_Array;
using Reference_Counted_Array::Release_Data;
using Reference_Counted_Array::Untyped_Array;

// ----------------------------------------------------------------------------
//
static void *initialize_numpy();

// ----------------------------------------------------------------------------
// Py_DECREF() and object when deleted.
// This is so SArray can use NumPy data without making a copy.
//
class Python_Decref : public Release_Data
{
public:
  Python_Decref(PyObject *object) { this->object = object; }
  virtual ~Python_Decref() { PyObject *tmp = object; object = NULL; Py_DECREF(tmp); }
  PyObject *python_object() const { return object; }
private:
  PyObject *object;
};

// ----------------------------------------------------------------------------
//
bool array_from_python(PyObject *array, int dim, Numeric_Array *na, bool allow_data_copy)
{
  initialize_numpy();

  PyArrayObject *a;
  if (PyArray_Check(array))
    {
      a = (PyArrayObject *) array;
      Py_XINCREF(array);
    }
  else if (allow_data_copy)
    a = (PyArrayObject *) PyArray_FromObject(array, NPY_NOTYPE, 0, 0);
  else
    {
      PyErr_SetString(PyExc_TypeError, "NumPy array required");
      return false;
    }

  if (a == (PyArrayObject *)0)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid array argument");
      return false;
    }

  if (dim == 0)
    dim = PyArray_NDIM(a);  // Accept any dimension.
  else if (PyArray_NDIM(a) != dim)
    {
      Py_DECREF((PyObject *) a);
      PyErr_Format(PyExc_TypeError, "Array must be %d-dimensional, got %d-dimensional", dim, PyArray_NDIM(a));
      return false;
    }

  Numeric_Array::Value_Type dtype;
  int type = PyArray_TYPE(a);
  switch ((NPY_TYPES) type)
    {
    case NPY_BOOL:  dtype = Numeric_Array::Unsigned_Char;   break;
    case NPY_UBYTE: dtype = Numeric_Array::Unsigned_Char;   break;
    case NPY_BYTE:  dtype = Numeric_Array::Signed_Char; break;
    case NPY_SHORT: dtype = Numeric_Array::Short_Int;   break;
    case NPY_USHORT:    dtype = Numeric_Array::Unsigned_Short_Int;  break;
    case NPY_INT:   dtype = Numeric_Array::Int;     break;
    case NPY_UINT:  dtype = Numeric_Array::Unsigned_Int;    break;
    case NPY_LONG:
      // Make 32-bit integers always be int instead of long on 32-bit machines.
      dtype = (sizeof(int) == sizeof(long) ? Numeric_Array::Int : Numeric_Array::Long_Int); break;
    case NPY_ULONG:
      dtype = (sizeof(int) == sizeof(long) ? Numeric_Array::Unsigned_Int : Numeric_Array::Unsigned_Long_Int);   break;
    case NPY_FLOAT: dtype = Numeric_Array::Float;       break;
    case NPY_DOUBLE:    dtype = Numeric_Array::Double;      break;
    default:
      PyErr_SetString(PyExc_TypeError, "Array argument has non-numeric values");
      return false;
    };

  int64_t *sizes = new int64_t[dim];
  for (int k = 0 ; k < dim ; ++k)
    sizes[k] = PyArray_DIM(a,k);

  //
  // NumPy strides are in bytes.
  // Numeric_Array strides are in elements.
  //
  int64_t *strides = new int64_t[dim];
  int element_size = PyArray_ITEMSIZE(a);
  for (int k = 0 ; k < dim ; ++k)
    strides[k] = PyArray_STRIDE(a,k) / element_size;

  void *data = PyArray_DATA(a);
  Release_Data *release = new Python_Decref((PyObject *)a);

  *na = Numeric_Array(dtype, dim, sizes, strides, data, release);

  delete [] strides;
  delete [] sizes;

  return true;
}

// ----------------------------------------------------------------------------
//
PyObject *array_python_source(const Untyped_Array &a, bool incref)
{
  const Release_Data *r = a.release_method();
  if (r == NULL)
    return NULL;
  const Python_Decref *p = dynamic_cast<const Python_Decref *>(r);
  if (p == NULL)
    return NULL;
  PyObject *na = p->python_object();
  if (incref)
    Py_INCREF(na);
  return na;
}

// ----------------------------------------------------------------------------
//
bool array_from_python(PyObject *array, int dim,
               Numeric_Array::Value_Type required_type, Numeric_Array *na,
               bool allow_data_copy)
{
  Numeric_Array a;
  if (!array_from_python(array, dim, &a, allow_data_copy))
    return false;

  if (a.value_type() == required_type)
    {
      *na = a;
      return true;
    }

  if (!allow_data_copy)
    {
      PyErr_Format(PyExc_TypeError, "Require array value type %s, got %s",
           Numeric_Array::value_type_name(required_type),
           Numeric_Array::value_type_name(a.value_type()));
    return false;
    }

  *na = a.as_type(required_type);

  return true;
}

// ----------------------------------------------------------------------------
//
bool python_array_to_c(PyObject *a, double *values, int64_t size)
{
  if (!PySequence_Check(a))
    {
      PyErr_SetString(PyExc_TypeError, "Array argument (1d float64) is not a sequence");
      return false;
    }

  if (PySequence_Size(a) != size)
    {
      PyErr_Format(PyExc_TypeError, "Incorrect array size, got %d, should be %d",
           PySequence_Size(a), size);
      return false;
    }

  for (int64_t k = 0 ; k < size ; ++k)
    {
      PyObject *e = PySequence_GetItem(a, k);
      if (!PyNumber_Check(e))
    {
      Py_DECREF(e);
      PyErr_SetString(PyExc_TypeError, "Array element is not a number");
      return false;
    }
      PyObject *f = PyNumber_Float(e);
      Py_DECREF(e);
      if (f == (PyObject *) 0)
    {
      PyErr_SetString(PyExc_TypeError, "Array element is not a float");
      return false;
    }
      values[k] = PyFloat_AsDouble(f);
      Py_DECREF(f);
    }
  return true;
}

// ----------------------------------------------------------------------------
//
bool python_array_to_c(PyObject *a, float *values, int64_t size)
{
  if (!PySequence_Check(a))
    {
      PyErr_SetString(PyExc_TypeError, "Array argument (1d float32) is not a sequence");
      return false;
    }

  if (PySequence_Size(a) != size)
    {
      PyErr_Format(PyExc_TypeError, "Incorrect array size, got %d, should be %d",
           PySequence_Size(a), size);
      return false;
    }

  for (int64_t k = 0 ; k < size ; ++k)
    {
      PyObject *e = PySequence_GetItem(a, k);
      if (!PyNumber_Check(e))
    {
      Py_DECREF(e);
      PyErr_SetString(PyExc_TypeError, "Array element is not a number");
      return false;
    }
      PyObject *f = PyNumber_Float(e);
      Py_DECREF(e);
      if (f == (PyObject *) 0)
    {
      PyErr_SetString(PyExc_TypeError, "Array element is not a float");
      return false;
    }
      double v = PyFloat_AsDouble(f);
      values[k] = static_cast<float>(v);
      Py_DECREF(f);
    }
  return true;
}

// ----------------------------------------------------------------------------
//
bool python_array_to_c(PyObject *a, float *values, int64_t size0, int64_t size1)
{
  initialize_numpy();       // required before using NumPy.

  PyObject *na = PyArray_ContiguousFromObject(a, NPY_DOUBLE, 2, 2);
  if (na == NULL)
    {
      PyErr_SetString(PyExc_TypeError, "Array argument (2d float32) is not a sequence");
      return false;
    }

  PyArrayObject *ao = reinterpret_cast<PyArrayObject *>(na);
  if (PyArray_DIM(ao,0) != size0 || PyArray_DIM(ao,1) != size1)
    {
      PyErr_Format(PyExc_TypeError, "Incorrect 2-D array size, got (%d,%d), expected (%d,%d)",
           PyArray_DIM(ao,0), PyArray_DIM(ao,1), size0, size1);
      return false;
    }

  int64_t n = size0 * size1;
  double *d = reinterpret_cast<double *>(PyArray_DATA(ao));
  for (int64_t k = 0 ; k < n ; ++k)
      values[k] = static_cast<float>(d[k]);

  Py_DECREF(na);
  return true;
}

// ----------------------------------------------------------------------------
//
bool python_array_to_c(PyObject *a, double *values, int64_t size0, int64_t size1)
{
  initialize_numpy();       // required before using NumPy.

  PyObject *na = PyArray_ContiguousFromObject(a, NPY_DOUBLE, 2, 2);
  if (na == NULL)
    {
      PyErr_SetString(PyExc_TypeError, "Array argument (2d float64) is not a sequence");
      return false;
    }

  PyArrayObject *ao = reinterpret_cast<PyArrayObject *>(na);
  if (PyArray_DIM(ao,0) != size0 || PyArray_DIM(ao,1) != size1)
    {
      PyErr_Format(PyExc_TypeError, "Incorrect 2-D array size, got (%d,%d), expected (%d,%d)",
           PyArray_DIM(ao,0), PyArray_DIM(ao,1), size0, size1);
      return false;
    }

  int64_t n = size0 * size1;
  double *d = reinterpret_cast<double *>(PyArray_DATA(ao));
  for (int64_t k = 0 ; k < n ; ++k)
      values[k] = d[k];

  Py_DECREF(na);
  return true;
}

// ----------------------------------------------------------------------------
//
bool python_array_to_c(PyObject *a, int *values, int64_t size)
{
  initialize_numpy();       // required before using NumPy.

  PyObject *na = PyArray_ContiguousFromObject(a, NPY_INT, 1, 1);
  if (na == NULL)
    {
      PyErr_SetString(PyExc_TypeError, "Array argument (1d int32) is not a sequence");
      return false;
    }

  PyArrayObject *ao = reinterpret_cast<PyArrayObject *>(na);
  if (PyArray_DIM(ao,0) != size)
    {
      PyErr_Format(PyExc_TypeError, "Incorrect 2-D array size, got %d, expected %d",
           PyArray_DIM(ao,0), size);
      return false;
    }

  int *d = reinterpret_cast<int *>(PyArray_DATA(ao));
  for (int64_t k = 0 ; k < size ; ++k)
      values[k] = d[k];

  Py_DECREF(na);
  return true;
}

// ----------------------------------------------------------------------------
//
bool float_2d_array_values(PyObject *farray, int64_t n2, float **f, int64_t *size)
{
  initialize_numpy();       // required before using NumPy.

  if (!PyArray_Check(farray))
    {
      PyErr_SetString(PyExc_TypeError, "argument is not a NumPy array");
      return false;
    }

  PyArrayObject *fa = static_cast<PyArrayObject *>(static_cast<void *>(farray));
  if (PyArray_TYPE(fa) != NPY_FLOAT)
    {
      PyErr_SetString(PyExc_TypeError, "NumPy array is not of type float");
      return false;
    }

  if (PyArray_NDIM(fa) != 2)
    {
      PyErr_SetString(PyExc_TypeError, "NumPy array is not 2 dimensional");
      return false;
    }

  if (PyArray_DIM(fa,1) != n2)
    {
      PyErr_Format(PyExc_TypeError, "NumPy array 2nd dimension is not size %d", n2);
      return false;
    }

  // Check if array is contiguous.
  if (PyArray_DIM(fa,0) > 0 &&
      (PyArray_STRIDE(fa,1) != static_cast<int>(sizeof(float)) ||
       PyArray_STRIDE(fa,0) != static_cast<int>(n2*sizeof(float))))
    {
      PyErr_SetString(PyExc_TypeError, "NumPy array is not contiguous");
      return false;
    }

  *f = static_cast<float *>(static_cast<void *>(PyArray_DATA(fa)));
  *size = n2*PyArray_DIM(fa,0);

  return true;
}

// ----------------------------------------------------------------------------
//
static const char *numpy_type_name(int type)
{
  const char *name = "unknown";
  switch (type)
    {
    case NPY_BOOL: name = "bool"; break;
    case NPY_BYTE: name = "byte"; break;
    case NPY_UBYTE: name = "ubyte"; break;
    case NPY_SHORT: name = "short"; break;
    case NPY_USHORT: name = "ushort"; break;
    case NPY_INT: name = "int"; break;
    case NPY_UINT: name = "uint"; break;
    case NPY_LONG: name = "long"; break;
    case NPY_ULONG: name = "ulong"; break;
    case NPY_LONGLONG: name = "longlong"; break;
    case NPY_ULONGLONG: name = "ulonglong"; break;
    case NPY_FLOAT: name = "float"; break;
    case NPY_DOUBLE: name = "double"; break;
    case NPY_LONGDOUBLE: name = "longdouble"; break;
    case NPY_CFLOAT: name = "cfloat"; break;
    case NPY_CDOUBLE: name = "cdouble"; break;
    case NPY_CLONGDOUBLE: name = "clongdouble"; break;
    case NPY_OBJECT: name = "object"; break;
    case NPY_STRING: name = "string"; break;
    case NPY_UNICODE: name = "unicode"; break;
    case NPY_VOID: name = "void"; break;
    default: break;
  }
  return name;
}

// ----------------------------------------------------------------------------
// Array is not initialized to zero.
//
static PyObject *allocate_python_array(int dim, int64_t *size, int type)
{
  npy_intp *sn = new npy_intp[dim];
  for (int i = 0 ; i < dim ; ++i)
    sn[i] = (npy_intp)size[i];

  PyObject *a = PyArray_SimpleNew(dim, sn, type);
  delete [] sn;
  if (a == NULL)
    {
      int64_t s = 1;
      for (int i = 0 ; i < dim ; ++i)
        s *= size[i];
      PyErr_Format(PyExc_MemoryError, "%s array allocation of size %ld, dimension %d, value type %d failed",
           numpy_type_name(type), s, dim, type);
      return NULL;
    }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const int *data, int64_t size)
{
  initialize_numpy();       // required before using NumPy.

  int64_t shape[1] = {size};
  PyObject *a = allocate_python_array(1, shape, NPY_INT);
  if (a) {
    int *py_data = (int *) PyArray_DATA((PyArrayObject *)a);
    for (int64_t k = 0 ; k < size ; ++k)
      py_data[k] = data[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const std::vector<int> &i)
{
  initialize_numpy();       // required before using NumPy.

  int64_t sz = i.size();
  int64_t dimensions[1] = {sz};
  PyObject *a = allocate_python_array(1, dimensions, NPY_INT);
  if (a) {
    int *py_data = (int *)PyArray_DATA((PyArrayObject *)a);
    for (int64_t k = 0 ; k < sz ; ++k)
       py_data[k] = i[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
// Returns an array of long which will be 32-bit on Windows,
// 64-bit on other platforms.
//
// TODO: Return 64-bit on Windows.  I will need to add NPY_INT64 support
// to Numeric_Array and the ability of template code to operate on that type.
//
PyObject *c_array_to_python(const std::vector<int64_t> &i)
{
  initialize_numpy();       // required before using NumPy.

  int64_t sz = i.size();
  int64_t dimensions[1] = {sz};
  PyObject *a = allocate_python_array(1, dimensions, NPY_LONG);
  if (a) {
    long *py_data = (long *)PyArray_DATA((PyArrayObject *)a);
    for (int64_t k = 0 ; k < sz ; ++k)
       py_data[k] = i[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const std::vector<int> &i, int64_t size0, int64_t size1)
{
  initialize_numpy();       // required before using NumPy.

  int64_t sz = i.size();
  int64_t dimensions[2] = {size0, size1};
  PyObject *a = allocate_python_array(2, dimensions, NPY_INT);
  if (a) {
    int *py_data = (int *)PyArray_DATA((PyArrayObject *)a);
    for (int64_t k = 0 ; k < sz ; ++k)
      py_data[k] = i[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const std::vector<float> &values)
{
  initialize_numpy();       // required before using NumPy.

  int64_t sz = values.size();
  int64_t dimensions[1] = {sz};
  PyObject *a = allocate_python_array(1, dimensions, NPY_FLOAT);
  if (a) {
    float *py_data = (float *)PyArray_DATA((PyArrayObject *)a);
    for (int64_t k = 0 ; k < sz ; ++k)
      py_data[k] = values[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const std::vector<float> &values, int64_t size0, int64_t size1)
{
  initialize_numpy();       // required before using NumPy.

  int64_t sz = values.size();
  int64_t dimensions[2] = {size0, size1};
  PyObject *a = allocate_python_array(2, dimensions, NPY_FLOAT);
  if (a) {
    float *py_data = (float *)PyArray_DATA((PyArrayObject *)a);
    for (int64_t k = 0 ; k < sz ; ++k)
      py_data[k] = values[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const float *values, int64_t size)
{
  initialize_numpy();       // required before using NumPy.

  int64_t shape[1] = {size};
  PyObject *a = allocate_python_array(1, shape, NPY_FLOAT);
  if (a) {
    float *py_data = (float *) PyArray_DATA((PyArrayObject *)a);
    for (int64_t k = 0 ; k < size ; ++k)
      py_data[k] = values[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
// TODO: Make this return a NumPy array.
//
PyObject *c_array_to_python(const double *values, int64_t size)
{
  initialize_numpy();       // required before using NumPy.

  int64_t shape[1] = {size};
  PyObject *a = allocate_python_array(1, shape, NPY_DOUBLE);
  if (a) {
    double *py_data = (double *) PyArray_DATA((PyArrayObject *)a);
    for (int64_t k = 0 ; k < size ; ++k)
      py_data[k] = values[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const float *data, int64_t size0, int64_t size1)
{
  initialize_numpy();       // required before using NumPy.

  int64_t shape[2] = {size0, size1};
  PyObject *a = allocate_python_array(2, shape, NPY_FLOAT);
  if (a) {
    float *py_data = (float *) PyArray_DATA((PyArrayObject *)a);
    int64_t size = size0 * size1;
    for (int64_t k = 0 ; k < size ; ++k)
      py_data[k] = data[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const double *data, int64_t size0, int64_t size1)
{
  initialize_numpy();       // required before using NumPy.

  int64_t shape[2] = {size0, size1};
  PyObject *a = allocate_python_array(2, shape, NPY_DOUBLE);
  if (a) {
    double *py_data = (double *) PyArray_DATA((PyArrayObject *)a);
    int64_t size = size0 * size1;
    for (int64_t k = 0 ; k < size ; ++k)
      py_data[k] = data[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const int *data, int64_t size0, int64_t size1)
{
  initialize_numpy();       // required before using NumPy.

  int64_t shape[2] = {size0, size1};
  PyObject *a = allocate_python_array(2, shape, NPY_INT);
  if (a) {
    int *py_data = (int *) PyArray_DATA((PyArrayObject *)a);
    int64_t size = size0 * size1;
    for (int64_t k = 0 ; k < size ; ++k)
      py_data[k] = data[k];
  }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_bool_array(int64_t size, unsigned char **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_BOOL);
  if (a && data)
    *data = (unsigned char *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_uint8_array(int64_t size, unsigned char **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_UINT8);
  if (a && data)
    *data = (unsigned char *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_uint8_array(int64_t size1, int64_t size2, unsigned char **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[2] = {size1, size2};
  PyObject *a = allocate_python_array(2, dimensions, NPY_UINT8);
  if (a && data)
    *data = (unsigned char *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_int_array(int64_t size, int **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_INT);
  if (a && data)
    *data = (int *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_int_array(int64_t size1, int64_t size2, int **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[2] = {size1, size2};
  PyObject *a = allocate_python_array(2, dimensions, NPY_INT);
  if (a && data)
    *data = (int *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_unsigned_int_array(int64_t size1, int64_t size2, int64_t size3, unsigned int **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[3] = {size1, size2, size3};
  PyObject *a = allocate_python_array(3, dimensions, NPY_UINT);
  if (a && data)
    *data = (unsigned int *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_float_array(int64_t size, float **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_FLOAT);
  if (a && data)
    *data = (float *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_float_array(int64_t size1, int64_t size2, float **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[2] = {size1, size2};
  PyObject *a = allocate_python_array(2, dimensions, NPY_FLOAT);
  if (a && data)
    *data = (float *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_float_array(int64_t size1, int64_t size2, int64_t size3, float **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[3] = {size1, size2, size3};
  PyObject *a = allocate_python_array(3, dimensions, NPY_FLOAT);
  if (a && data)
    *data = (float *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_double_array(int64_t size, double **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_DOUBLE);
  if (a && data)
    *data = (double *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_double_array(int64_t size1, int64_t size2, double **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[2] = {size1, size2};
  PyObject *a = allocate_python_array(2, dimensions, NPY_DOUBLE);
  if (a && data)
    *data = (double *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_double_array(int64_t size1, int64_t size2, int64_t size3, double **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[3] = {size1, size2, size3};
  PyObject *a = allocate_python_array(3, dimensions, NPY_DOUBLE);
  if (a && data)
    *data = (double *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_voidp_array(int64_t size, void ***data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_UINTP);
  if (a && data)
    *data = (void **)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_object_array(int64_t size, PyObject **data)
{
  initialize_numpy();       // required before using NumPy.

  int64_t dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_OBJECT);
  if (a && data)
    *data = (PyObject *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_1d_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  return array_from_python(arg, 1, na) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_2d_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  return array_from_python(arg, 2, na) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_2d_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  return array_from_python(arg, 2, na, false) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_3d_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  return array_from_python(arg, 3, na) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_3d_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  return array_from_python(arg, 3, na, false) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  return array_from_python(arg, 0, na) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  return array_from_python(arg, 0, na, false) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_array(PyObject *arg, void *farray)
{
  Numeric_Array na;
  if (array_from_python(arg, 0, Numeric_Array::Float, &na))
    {
      *static_cast<FArray *>(farray) = na;
      return 1;
    }
  return 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_4d_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  return array_from_python(arg, 4, na, false) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_bool(PyObject *arg, void *b)
{
  if (!PyBool_Check(arg))
    {
      PyErr_SetString(PyExc_TypeError,
              "boolean argument must be True or False");
      return 0;
    }
  *(bool *)b = (arg == Py_True);
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_int_3_array(PyObject *arg, void *i3)
{
  return python_array_to_c(arg, static_cast<int*>(i3), 3) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_3_array(PyObject *arg, void *f3)
{
  return python_array_to_c(arg, static_cast<float*>(f3), 3) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_4_array(PyObject *arg, void *f4)
{
  return python_array_to_c(arg, static_cast<float*>(f4), 4) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_double_3_array(PyObject *arg, void *f3)
{
  return python_array_to_c(arg, static_cast<double*>(f3), 3) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
static int parse_float_n(PyObject *arg, void *farray, bool allow_copy, bool f64)
{
  Numeric_Array::Value_Type ftype = (f64 ? Numeric_Array::Double : Numeric_Array::Float);
  Numeric_Array v;
  if (!array_from_python(arg, 1, ftype, &v, allow_copy))
    return 0;
  if (f64)
    *static_cast<DArray*>(farray) = static_cast<DArray>(v);
  else
    *static_cast<FArray*>(farray) = static_cast<FArray>(v);
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_n_array(PyObject *arg, void *farray)
  { return parse_float_n(arg, farray, true, false); }
extern "C" int parse_writable_float_n_array(PyObject *arg, void *farray)
  { return parse_float_n(arg, farray, false, false); }
extern "C" int parse_double_n_array(PyObject *arg, void *farray)
  { return parse_float_n(arg, farray, true, true); }
extern "C" int parse_writable_double_n_array(PyObject *arg, void *farray)
  { return parse_float_n(arg, farray, false, true); }

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_3x3_array(PyObject *arg, void *f33)
{
  return python_array_to_c(arg, static_cast<float*>(f33), 3, 3) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_double_3x3_array(PyObject *arg, void *d33)
{
  return python_array_to_c(arg, static_cast<double*>(d33), 3, 3) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_3x4_array(PyObject *arg, void *f34)
{
  return python_array_to_c(arg, static_cast<float*>(f34), 3, 4) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_double_3x4_array(PyObject *arg, void *d34)
{
  return python_array_to_c(arg, static_cast<double*>(d34), 3, 4) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
static int parse_nm_array(PyObject *arg, Numeric_Array::Value_Type vtype, int64_t m, bool allow_copy,
			  Numeric_Array &v)
{
  if (!array_from_python(arg, 0, vtype, &v, allow_copy))
    return 0;

  if (v.dimension() == 1 && v.size() == 0)
    {
      int64_t size[2] = {0,m};
      v = Numeric_Array(vtype, 2, size);
    }
  if (v.dimension() != 2)
    {
      PyErr_Format(PyExc_TypeError, "Array must be 2 dimensional, got %d dimensional", v.dimension());
      return 0;
    }
  if (v.size(1) != m)
    {
      PyErr_Format(PyExc_TypeError, "Second array dimension must have size %d.", m);
      return 0;
    }
  return 1;
}

// ----------------------------------------------------------------------------
//
static int parse_float_nm(PyObject *arg, int64_t m, void *farray, bool allow_copy, bool f64)
{
  Numeric_Array::Value_Type ftype = (f64 ? Numeric_Array::Double : Numeric_Array::Float);
  Numeric_Array v;
  int s = parse_nm_array(arg, ftype, m, allow_copy, v);
  if (s)
    {
      if (f64)
        *static_cast<DArray*>(farray) = static_cast<DArray>(v);
      else
        *static_cast<FArray*>(farray) = static_cast<FArray>(v);
    }
  return s;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_contiguous_float_4x4_array(PyObject *arg, void *farray)
{
  Numeric_Array v;
  bool s = array_from_python(arg, 2, Numeric_Array::Float, &v, false);
  if (s)
    {
      if (v.size(0) != 4 || v.size(1) != 4)
	{
	  PyErr_Format(PyExc_TypeError, "Require array size 4x4, got %s", v.size_string().c_str());
	  s = false;
	}
      else if (!v.is_contiguous())
	{
	  PyErr_Format(PyExc_TypeError, "Require contiguous 4x4 array");
	  s = false;
	}
      else
	*static_cast<FArray*>(farray) = static_cast<FArray>(v);
    }
  return (s ? 1 : 0);
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_contiguous_float_n44_array(PyObject *arg, void *farray)
{
  Numeric_Array v;
  bool s = array_from_python(arg, 3, Numeric_Array::Float, &v, false);
  if (s)
    {
      if (v.size(1) != 4 || v.size(2) != 4)
	{
	  PyErr_Format(PyExc_TypeError, "Require array size n x 4 x 4, got %s", v.size_string().c_str());
	  s = false;
	}
      else if (!v.is_contiguous())
	{
	  PyErr_Format(PyExc_TypeError, "Require contiguous n x 4 x 4 array");
	  s = false;
	}
      else
	*static_cast<FArray*>(farray) = static_cast<FArray>(v);
    }
  return (s ? 1 : 0);
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_contiguous_double_3x4_array(PyObject *arg, void *darray)
{
  Numeric_Array v;
  bool s = array_from_python(arg, 2, Numeric_Array::Double, &v, false);
  if (s)
    {
      if (v.size(0) != 3 || v.size(1) != 4)
	{
	  PyErr_Format(PyExc_TypeError, "Require array size 3x4, got %s", v.size_string().c_str());
	  s = false;
	}
      else if (!v.is_contiguous())
	{
	  PyErr_Format(PyExc_TypeError, "Require contiguous 3x4 array");
	  s = false;
	}
      else
	*static_cast<DArray*>(darray) = static_cast<DArray>(v);
    }
  return (s ? 1 : 0);
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_contiguous_double_n34_array(PyObject *arg, void *darray)
{
  Numeric_Array v;
  bool s = array_from_python(arg, 3, Numeric_Array::Double, &v, false);
  if (s)
    {
      if (v.size(1) != 3 || v.size(2) != 4)
	{
	  PyErr_Format(PyExc_TypeError, "Require array size n x 3 x 4, got %s",
		       v.size_string().c_str());
	  s = false;
	}
      else if (!v.is_contiguous())
	{
	  PyErr_Format(PyExc_TypeError, "Require contiguous n x 3 x 4 array");
	  s = false;
	}
      else
	*static_cast<DArray*>(darray) = static_cast<DArray>(v);
    }
  return (s ? 1 : 0);
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_n2_array(PyObject *arg, void *farray)
  { return parse_float_nm(arg, 2, farray, true, false); }
extern "C" int parse_float_n3_array(PyObject *arg, void *farray)
  { return parse_float_nm(arg, 3, farray, true, false); }
extern "C" int parse_writable_float_n3_array(PyObject *arg, void *farray)
  { return parse_float_nm(arg, 3, farray, false, false); }
extern "C" int parse_double_n3_array(PyObject *arg, void *darray)
  { return parse_float_nm(arg, 3, darray, true, true); }
extern "C" int parse_writable_double_n3_array(PyObject *arg, void *darray)
  { return parse_float_nm(arg, 3, darray, false, true); }
extern "C" int parse_float_n4_array(PyObject *arg, void *farray)
  { return parse_float_nm(arg, 4, farray, true, false); }
extern "C" int parse_writable_float_n4_array(PyObject *arg, void *farray)
  { return parse_float_nm(arg, 4, farray, false, false); }
extern "C" int parse_writable_float_n9_array(PyObject *arg, void *farray)
  { return parse_float_nm(arg, 9, farray, false, false); }

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_2d_array(PyObject *arg, void *farray)
{
  Numeric_Array v;
  if (!array_from_python(arg, 2, Numeric_Array::Float, &v, true))
    return 0;
  *static_cast<FArray*>(farray) = static_cast<FArray>(v);

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_float_2d_array(PyObject *arg, void *farray)
{
  Numeric_Array v;
  if (!array_from_python(arg, 2, Numeric_Array::Float, &v, false))
    return 0;
  *static_cast<FArray*>(farray) = static_cast<FArray>(v);

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_float_3d_array(PyObject *arg, void *farray)
{
  Numeric_Array v;
  if (!array_from_python(arg, 3, Numeric_Array::Float, &v, false))
    return 0;
  *static_cast<FArray*>(farray) = static_cast<FArray>(v);

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_int_2d_array(PyObject *arg, void *iarray)
{
  Numeric_Array v;
  if (!array_from_python(arg, 2, Numeric_Array::Int, &v, true))
    return 0;
  *static_cast<IArray*>(iarray) = static_cast<IArray>(v);

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_int_2d_array(PyObject *arg, void *iarray)
{
  Numeric_Array v;
  if (!array_from_python(arg, 2, Numeric_Array::Int, &v, false))
    return 0;
  *static_cast<IArray*>(iarray) = static_cast<IArray>(v);

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_uint8_n_array(PyObject *arg, void *barray)
{
  Numeric_Array v;
  bool allow_copy = true;
  if (!array_from_python(arg, 1, Numeric_Array::Unsigned_Char, &v, allow_copy))
    return 0;
  *static_cast<BArray*>(barray) = static_cast<BArray>(v);
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_uint8_n_array(PyObject *arg, void *barray)
{
  Numeric_Array v;
  bool allow_copy = false;
  if (!array_from_python(arg, 1, Numeric_Array::Unsigned_Char, &v, allow_copy))
    return 0;
  *static_cast<BArray*>(barray) = static_cast<BArray>(v);
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_uint8_n2_array(PyObject *arg, void *barray)
{
  Numeric_Array v;
  int s = parse_nm_array(arg, Numeric_Array::Unsigned_Char, 2, false, v);
  if (s)
    *static_cast<BArray*>(barray) = static_cast<BArray>(v);
  return s;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_uint8_n3_array(PyObject *arg, void *barray)
{
  Numeric_Array v;
  int s = parse_nm_array(arg, Numeric_Array::Unsigned_Char, 3, false, v);
  if (s)
    *static_cast<BArray*>(barray) = static_cast<BArray>(v);
  return s;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_uint8_n4_array(PyObject *arg, void *barray)
{
  Numeric_Array v;
  int s = parse_nm_array(arg, Numeric_Array::Unsigned_Char, 4, false, v);
  if (s)
    *static_cast<BArray*>(barray) = static_cast<BArray>(v);
  return s;
}

// ----------------------------------------------------------------------------
//
static int parse_int_nm(PyObject *arg, int64_t m, void *iarray, bool allow_copy)
{
  Numeric_Array v;
  if (!array_from_python(arg, 0, &v, allow_copy))
    return 0;

  if (v.dimension() == 1 && v.size() == 0)
    {
      int64_t size[2] = {0,m};
      v = Numeric_Array(Numeric_Array::Int, 2, size);
    }
  if (v.dimension() != 2)
    {
      PyErr_Format(PyExc_TypeError, "Array must be 2 dimensional, got %d dimensional", v.dimension());
      return 0;
    }
  if (v.value_type() == Numeric_Array::Long_Int && allow_copy)
    {
      IArray vi = IArray(v.dimension(), v.sizes());
      vi.set(Reference_Counted_Array::Array<long int>(v));
      v = Numeric_Array(Numeric_Array::Int, vi);
    }
  if (v.value_type() != Numeric_Array::Int)
    {
      PyErr_Format(PyExc_TypeError, "array type must be int or long int, got %s",
           Numeric_Array::value_type_name(v.value_type()));
      return 0;
    }
  if (v.size(1) != m)
    {
      PyErr_Format(PyExc_TypeError, "Second array dimension must have size %d, got %d", m, v.size(1));
      return 0;
    }
  *static_cast<IArray*>(iarray) = static_cast<IArray>(v);

  return 1;
}

// ----------------------------------------------------------------------------
//
static int parse_int_array(PyObject *arg, void *iarray, bool allow_copy)
{
  Numeric_Array v;
  if (!array_from_python(arg, 1, Numeric_Array::Int, &v, allow_copy))
    return 0;
  *static_cast<IArray*>(iarray) = static_cast<IArray>(v);

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_int_n_array(PyObject *arg, void *iarray)
  { return parse_int_array(arg, iarray, true); }
extern "C" int parse_writable_int_n_array(PyObject *arg, void *iarray)
  { return parse_int_array(arg, iarray, false); }

// ----------------------------------------------------------------------------
//
extern "C" int parse_int_n2_array(PyObject *arg, void *iarray)
  { return parse_int_nm(arg, 2, iarray, true); }
extern "C" int parse_int_n3_array(PyObject *arg, void *iarray)
  { return parse_int_nm(arg, 3, iarray, true); }
extern "C" int parse_writable_int_n3_array(PyObject *arg, void *iarray)
  { return parse_int_nm(arg, 3, iarray, false); }

// ----------------------------------------------------------------------------
// Convert 1-d string array with fixed length strings to 2-d character array.
//
extern "C" int parse_string_array(PyObject *array, void *carray)
{
  initialize_numpy();
  
  if (!PyArray_Check(array))
    {
      PyErr_SetString(PyExc_TypeError, "argument is not a NumPy array");
      return 0;
    }

  PyArrayObject *a = (PyArrayObject *) array;

  int dim = PyArray_NDIM(a);
  if (dim != 1)
    {
      PyErr_SetString(PyExc_TypeError, "argument is not a 1-d NumPy array");
      return 0;
    }

  int type = PyArray_TYPE(a);
  if (type != NPY_STRING)
    {
      PyErr_SetString(PyExc_TypeError, "argument is not a NumPy string array");
      return 0;
    }

  Numeric_Array::Value_Type dtype = Numeric_Array::Char;

  int64_t *sizes = new int64_t[dim+1];
  for (int k = 0 ; k < dim ; ++k)
    sizes[k] = PyArray_DIM(a,k);
  sizes[dim] = PyArray_ITEMSIZE(a);

  //
  // NumPy strides are in bytes.
  // Numeric_Array strides are in elements.
  //
  int64_t *strides = new int64_t[dim+1];
  for (int k = 0 ; k < dim ; ++k)
    strides[k] = PyArray_STRIDE(a,k);
  strides[dim] = 1;

  void *data = PyArray_DATA(a);
  Py_XINCREF(array);
  Release_Data *release = new Python_Decref((PyObject *)a);

  Numeric_Array na(dtype, dim+1, sizes, strides, data, release);
  *static_cast<Reference_Counted_Array::Array<char> *>(carray) = Reference_Counted_Array::Array<char>(na);

  delete [] strides;
  delete [] sizes;

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_voidp(PyObject *arg, void **p)
{
  *p = PyLong_AsVoidPtr(arg);
  return (PyErr_Occurred() == NULL) ? 1 : 0;
}


// ----------------------------------------------------------------------------
//
PyObject *python_tuple(PyObject *o1, PyObject *o2)
{
  PyObject *pair = PyTuple_New(2);
  PyTuple_SetItem(pair, 0, o1);
  PyTuple_SetItem(pair, 1, o2);
  return pair;
}

// ----------------------------------------------------------------------------
//
PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3)
{
  PyObject *t = PyTuple_New(3);
  PyTuple_SetItem(t, 0, o1);
  PyTuple_SetItem(t, 1, o2);
  PyTuple_SetItem(t, 2, o3);
  return t;
}

// ----------------------------------------------------------------------------
//
PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3, PyObject *o4)
{
  PyObject *t = PyTuple_New(4);
  PyTuple_SetItem(t, 0, o1);
  PyTuple_SetItem(t, 1, o2);
  PyTuple_SetItem(t, 2, o3);
  PyTuple_SetItem(t, 3, o4);
  return t;
}

// ----------------------------------------------------------------------------
//
PyObject *python_tuple(PyObject *o1, PyObject *o2, PyObject *o3, PyObject *o4, PyObject *o5)
{
  PyObject *t = PyTuple_New(5);
  PyTuple_SetItem(t, 0, o1);
  PyTuple_SetItem(t, 1, o2);
  PyTuple_SetItem(t, 2, o3);
  PyTuple_SetItem(t, 3, o4);
  PyTuple_SetItem(t, 4, o5);
  return t;
}

// ----------------------------------------------------------------------------
//
PyObject *python_none()
{
  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
PyObject *python_bool(bool b)
{
  PyObject *bpy = (b ? Py_True : Py_False);
  Py_INCREF(bpy);
  return bpy;
}

// ----------------------------------------------------------------------------
//
PyObject *python_voidp(void *p)
{
  return PyLong_FromVoidPtr(p);
}

// ----------------------------------------------------------------------------
// Need to call NumPy import_array() before using NumPy routines.
//
static void *initialize_numpy()
{
  static bool first_call = true;
  if (first_call)
    {
      first_call = false;
      import_array1(NULL);
    }
  return NULL;
}

// ----------------------------------------------------------------------------
//
static std::string int64_string(int64_t i)
{
  std::ostringstream string;
  string << i;
  return string.str();
}

// ----------------------------------------------------------------------------
//
bool check_array_size(FArray &a, int64_t n, int64_t m, bool require_contiguous)
{
  if (a.size(0) != n)
    {
      PyErr_Format(PyExc_TypeError, "Array size %s does not match other array argument size %s",
		   a.size_string(0).c_str(), int64_string(n).c_str());
      return false;
    }
  if (a.size(1) != m)
    {
      PyErr_Format(PyExc_TypeError, "The 2nd dimension of array must have size %d, got %d", m, a.size(1));
      return false;
    }
  if (require_contiguous && !a.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Array is non-contiguous");
      return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
bool check_array_size(FArray &a, int64_t n, bool require_contiguous)
{
  if (a.size(0) != n)
    {
      PyErr_Format(PyExc_TypeError, "Array size %s does not match other array argument size %s",
		   a.size_string(0).c_str(), int64_string(n).c_str());
      return false;
    }
  if (require_contiguous && !a.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Array is non-contiguous");
      return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
PyObject *resized_2d_array(PyObject *array, int64_t size0, int64_t size1)
{
  npy_intp dim[2] = {size0, size1};
  PyArray_Dims shape;
  shape.ptr = dim;
  shape.len = 2;
  PyArrayObject *a = reinterpret_cast<PyArrayObject *>(array);
  PyObject *ra = PyArray_Newshape(a, &shape, NPY_KEEPORDER);
  return ra;
}
