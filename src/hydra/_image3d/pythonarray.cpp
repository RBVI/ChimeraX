// ----------------------------------------------------------------------------
//
// #include <iostream>			// use std::cerr for debugging
#include <Python.h>			// use Py_DECREF()

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>		// use PyArray_*(), NPY_*

#include <stdexcept>			// use std::runtime_error
#include <sstream>			// use std::ostringstream

#include "pythonarray.h"
#include "rcarray.h"			// use Numeric_Array, Release_Data

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
  virtual ~Python_Decref() { Py_DECREF(this->object); object = (PyObject *) 0; }
  PyObject *python_object() const { return object; }
private:
  PyObject *object;
};

// ----------------------------------------------------------------------------
//
Numeric_Array array_from_python(PyObject *array, int dim, bool allow_data_copy)
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
    throw std::runtime_error("NumPy array required");

  if (a == (PyArrayObject *)0)
    {
      PyErr_Clear();
      throw std::runtime_error("Invalid array argument");
    }

  if (dim == 0)
    dim = PyArray_NDIM(a);	// Accept any dimension.
  else if (PyArray_NDIM(a) != dim)
    {
      Py_DECREF((PyObject *) a);
      std::ostringstream msg;
      msg << "Array must be " << dim << "-dimensional, got "
	  << PyArray_NDIM(a) << "-dimensional" << std::endl;
      throw std::runtime_error(msg.str());
    }

  Numeric_Array::Value_Type dtype;
  int type = PyArray_TYPE(a);
  switch ((NPY_TYPES) type)
    {
    case NPY_CHAR:	dtype = Numeric_Array::Char;		break;
    case NPY_UBYTE:	dtype = Numeric_Array::Unsigned_Char;	break;
    case NPY_BYTE:	dtype = Numeric_Array::Signed_Char;	break;
    case NPY_SHORT:	dtype = Numeric_Array::Short_Int;	break;
    case NPY_USHORT:	dtype = Numeric_Array::Unsigned_Short_Int;	break;
    case NPY_INT:	dtype = Numeric_Array::Int;		break;
    case NPY_UINT:	dtype = Numeric_Array::Unsigned_Int;	break;
    case NPY_LONG:
      // Make 32-bit integers always be int instead of long on 32-bit machines.
      dtype = (sizeof(int) == sizeof(long) ? Numeric_Array::Int : Numeric_Array::Long_Int);	break;
    case NPY_ULONG:
      dtype = (sizeof(int) == sizeof(long) ? Numeric_Array::Unsigned_Int : Numeric_Array::Unsigned_Long_Int);	break;
    case NPY_FLOAT:	dtype = Numeric_Array::Float;		break;
    case NPY_DOUBLE:	dtype = Numeric_Array::Double;		break;
    default:
      throw std::runtime_error("Array argument has non-numeric values");
    };

  int *sizes = new int[dim];
  for (int k = 0 ; k < dim ; ++k)
    sizes[k] = PyArray_DIM(a,k);

  //
  // NumPy strides are in bytes.
  // Numeric_Array strides are in elements.
  //
  long *strides = new long[dim];
  int element_size = PyArray_ITEMSIZE(a);
  for (int k = 0 ; k < dim ; ++k)
    strides[k] = PyArray_STRIDE(a,k) / element_size;

  void *data = PyArray_DATA(a);
  Release_Data *release = new Python_Decref((PyObject *)a);

  Numeric_Array na(dtype, dim, sizes, strides, data, release);

  delete [] strides;
  delete [] sizes;

  return na;
}

// ----------------------------------------------------------------------------
//
PyObject *array_python_source(const Untyped_Array &a)
{
  const Release_Data *r = a.release_method();
  if (r == NULL)
    return NULL;
  const Python_Decref *p = dynamic_cast<const Python_Decref *>(r);
  if (p == NULL)
    return NULL;
  PyObject *na = p->python_object();
  Py_INCREF(na);
  return na;
}

// ----------------------------------------------------------------------------
//
Numeric_Array array_from_python(PyObject *array, int dim,
				Numeric_Array::Value_Type required_type,
				bool allow_data_copy)
{
  Numeric_Array a = array_from_python(array, dim, allow_data_copy);
  if (a.value_type() == required_type)
    return a;

  if (allow_data_copy)
    a = a.as_type(required_type);
  else
    {
      std::ostringstream msg;
      msg << "Require array value type "
	  << Numeric_Array::value_type_name(required_type)
	  << ", got "
	  << Numeric_Array::value_type_name(a.value_type())
	  << std::endl;
      throw std::runtime_error(msg.str());
    }

  return a;
}

// ----------------------------------------------------------------------------
//
void python_array_to_c(PyObject *a, double *values, int size)
{
  if (!PySequence_Check(a))
    throw std::runtime_error("Array argument is not a sequence");

  if (PySequence_Size(a) != size)
    {
      std::ostringstream msg;
      msg << "Incorrect array size, got " << PySequence_Size(a)
	  << ", should be " << size << std::endl;
      throw std::runtime_error(msg.str());
    }

  for (int k = 0 ; k < size ; ++k)
    {
      PyObject *e = PySequence_GetItem(a, k);
      if (!PyNumber_Check(e))
	{
	  Py_DECREF(e);
	  throw std::runtime_error("Array element is not a number");
	}
      PyObject *f = PyNumber_Float(e);
      Py_DECREF(e);
      if (f == (PyObject *) 0)
	throw std::runtime_error("Array element is not a float");
      values[k] = PyFloat_AsDouble(f);
      Py_DECREF(f);
    }
}

// ----------------------------------------------------------------------------
//
void python_array_to_c(PyObject *a, float *values, int size)
{
  if (!PySequence_Check(a))
    throw std::runtime_error("Array argument is not a sequence");

  if (PySequence_Size(a) != size)
    {
      std::ostringstream msg;
      msg << "Incorrect array size, got " << PySequence_Size(a)
	  << ", should be " << size << std::endl;
      throw std::runtime_error(msg.str());
    }

  for (int k = 0 ; k < size ; ++k)
    {
      PyObject *e = PySequence_GetItem(a, k);
      if (!PyNumber_Check(e))
	{
	  Py_DECREF(e);
	  throw std::runtime_error("Array element is not a number");
	}
      PyObject *f = PyNumber_Float(e);
      Py_DECREF(e);
      if (f == (PyObject *) 0)
	throw std::runtime_error("Array element is not a float");
      double v = PyFloat_AsDouble(f);
      values[k] = static_cast<float>(v);
      Py_DECREF(f);
    }
}

// ----------------------------------------------------------------------------
//
void python_array_to_c(PyObject *a, float *values, int size0, int size1)
{
  initialize_numpy();		// required before using NumPy.

  PyObject *na = PyArray_ContiguousFromObject(a, NPY_DOUBLE, 2, 2);
  if (na == NULL)
    return;

  PyArrayObject *ao = reinterpret_cast<PyArrayObject *>(na);
  if (PyArray_DIM(ao,0) != size0 || PyArray_DIM(ao,1) != size1)
    {
      std::ostringstream msg;
      msg << "Incorrect 2-D array size, got ("
	  << PyArray_DIM(ao,0) << ", " << PyArray_DIM(ao,1) << "), "
	  << "expected (" << size0 << ", " << size1 << ")" << std::endl;
      throw std::runtime_error(msg.str());
    }

  int n = size0 * size1;
  double *d = reinterpret_cast<double *>(PyArray_DATA(ao));
  for (int k = 0 ; k < n ; ++k)
      values[k] = static_cast<float>(d[k]);

  Py_DECREF(na);
}

// ----------------------------------------------------------------------------
//
void python_array_to_c(PyObject *a, double *values, int size0, int size1)
{
  initialize_numpy();		// required before using NumPy.

  PyObject *na = PyArray_ContiguousFromObject(a, NPY_DOUBLE, 2, 2);
  if (na == NULL)
    return;

  PyArrayObject *ao = reinterpret_cast<PyArrayObject *>(na);
  if (PyArray_DIM(ao,0) != size0 || PyArray_DIM(ao,1) != size1)
    {
      std::ostringstream msg;
      msg << "Incorrect 2-D array size, got ("
	  << PyArray_DIM(ao,0) << ", " << PyArray_DIM(ao,1) << "), "
	  << "expected (" << size0 << ", " << size1 << ")" << std::endl;
      throw std::runtime_error(msg.str());
    }

  int n = size0 * size1;
  double *d = reinterpret_cast<double *>(PyArray_DATA(ao));
  for (int k = 0 ; k < n ; ++k)
      values[k] = d[k];

  Py_DECREF(na);
}

// ----------------------------------------------------------------------------
//
void python_array_to_c(PyObject *a, int *values, int size)
{
  initialize_numpy();		// required before using NumPy.

  PyObject *na = PyArray_ContiguousFromObject(a, NPY_INT, 1, 1);
  if (na == NULL)
    return;

  PyArrayObject *ao = reinterpret_cast<PyArrayObject *>(na);
  if (PyArray_DIM(ao,0) != size)
    {
      std::ostringstream msg;
      msg << "Incorrect 1-D array size, got "
	  << PyArray_DIM(ao,0) << ", " << "expected " << size << std::endl;
      throw std::runtime_error(msg.str());
    }

  int *d = reinterpret_cast<int *>(PyArray_DATA(ao));
  for (int k = 0 ; k < size ; ++k)
      values[k] = d[k];

  Py_DECREF(na);
}

// ----------------------------------------------------------------------------
//
bool float_2d_array_values(PyObject *farray, int n2, float **f, int *size)
{
  initialize_numpy();		// required before using NumPy.

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
  if (PyArray_STRIDE(fa,1) != static_cast<int>(sizeof(float)) ||
      PyArray_STRIDE(fa,0) != static_cast<int>(n2*sizeof(float)))
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
static PyObject *allocate_python_array(int dim, int *size, int type)
{
  npy_intp *sn = new npy_intp[dim];
  for (int i = 0 ; i < dim ; ++i)
    sn[i] = (npy_intp)size[i];

  PyObject *a = PyArray_SimpleNew(dim, sn, type);
  delete [] sn;
  if (a == NULL)
    {
      std::ostringstream msg;
      msg << numpy_type_name(type) << " array allocation of size (";
      for (int a = 0 ; a < dim ; ++a)
	msg << size[a] << (a < dim-1 ? ", " : "");
      msg << ") failed " << std::endl;
      throw std::runtime_error(msg.str());
    }
  return a;
}

// ----------------------------------------------------------------------------
// Array is not initialized to zero.
//
static PyObject *allocate_python_array(int dim, int *size, PyArray_Descr *dtype)
{
  npy_intp *sn = new npy_intp[dim];
  for (int i = 0 ; i < dim ; ++i)
    sn[i] = (npy_intp)size[i];

  PyObject *a = PyArray_SimpleNewFromDescr(dim, sn, dtype);
  delete [] sn;
  if (a == NULL)
    {
      std::ostringstream msg;
      msg << "Array allocation of size (";
      for (int a = 0 ; a < dim ; ++a)
	msg << size[a] << (a < dim-1 ? ", " : "");
      msg << ") failed " << std::endl;
      throw std::runtime_error(msg.str());
    }
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const int *data, int size)
{
  initialize_numpy();		// required before using NumPy.

  int shape[1] = {size};
  PyObject *a = allocate_python_array(1, shape, NPY_INT);
  int *py_data = (int *) PyArray_DATA((PyArrayObject *)a);
  for (int k = 0 ; k < size ; ++k)
    py_data[k] = data[k];

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const std::vector<int> &i)
{
  initialize_numpy();		// required before using NumPy.

  int sz = i.size();
  int dimensions[1] = {sz};
  PyObject *a = allocate_python_array(1, dimensions, NPY_INT);
  int *py_data = (int *)PyArray_DATA((PyArrayObject *)a);
  for (int k = 0 ; k < sz ; ++k)
    py_data[k] = i[k];
  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const float *values, int size)
{
  initialize_numpy();		// required before using NumPy.

  int shape[1] = {size};
  PyObject *a = allocate_python_array(1, shape, NPY_FLOAT);
  float *py_data = (float *) PyArray_DATA((PyArrayObject *)a);
  for (int k = 0 ; k < size ; ++k)
    py_data[k] = values[k];

  return a;
}

// ----------------------------------------------------------------------------
// TODO: Make this return a NumPy array.
//
PyObject *c_array_to_python(const double *values, int size)
{
  initialize_numpy();		// required before using NumPy.

  int shape[1] = {size};
  PyObject *a = allocate_python_array(1, shape, NPY_DOUBLE);
  double *py_data = (double *) PyArray_DATA((PyArrayObject *)a);
  for (int k = 0 ; k < size ; ++k)
    py_data[k] = values[k];

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const float *data, int size0, int size1)
{
  initialize_numpy();		// required before using NumPy.

  int shape[2] = {size0, size1};
  PyObject *a = allocate_python_array(2, shape, NPY_FLOAT);
  float *py_data = (float *) PyArray_DATA((PyArrayObject *)a);
  int size = size0 * size1;
  for (int k = 0 ; k < size ; ++k)
    py_data[k] = data[k];

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const double *data, int size0, int size1)
{
  initialize_numpy();		// required before using NumPy.

  int shape[2] = {size0, size1};
  PyObject *a = allocate_python_array(2, shape, NPY_DOUBLE);
  double *py_data = (double *) PyArray_DATA((PyArrayObject *)a);
  int size = size0 * size1;
  for (int k = 0 ; k < size ; ++k)
    py_data[k] = data[k];

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *c_array_to_python(const int *data, int size0, int size1)
{
  initialize_numpy();		// required before using NumPy.

  int shape[2] = {size0, size1};
  PyObject *a = allocate_python_array(2, shape, NPY_INT);
  int *py_data = (int *) PyArray_DATA((PyArrayObject *)a);
  int size = size0 * size1;
  for (int k = 0 ; k < size ; ++k)
    py_data[k] = data[k];

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_uint8_array(int size, unsigned char **data)
{
  initialize_numpy();		// required before using NumPy.

  int dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_UINT8);
  if (data)
    *data = (unsigned char *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_char_array(int size1, int size2, char **data)
{
  initialize_numpy();		// required before using NumPy.

  int dimensions[2] = {size1, size2};
  PyObject *a = allocate_python_array(2, dimensions, NPY_CHAR);
  if (data)
    *data = (char *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_string_array(int size, int string_length, char **data)
{
  initialize_numpy();		// required before using NumPy.

  PyArray_Descr *d = PyArray_DescrNewFromType(NPY_CHAR);
  d->elsize = string_length;
  int dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, d);
  if (data)
    *data = (char *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_int_array(int size, int **data)
{
  initialize_numpy();		// required before using NumPy.

  int dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_INT);
  if (data)
    *data = (int *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_int_array(int size1, int size2, int **data)
{
  initialize_numpy();		// required before using NumPy.

  int dimensions[2] = {size1, size2};
  PyObject *a = allocate_python_array(2, dimensions, NPY_INT);
  if (data)
    *data = (int *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_unsigned_int_array(int size1, int size2, int size3, unsigned int **data)
{
  initialize_numpy();		// required before using NumPy.

  int dimensions[3] = {size1, size2, size3};
  PyObject *a = allocate_python_array(3, dimensions, NPY_UINT);
  if (data)
    *data = (unsigned int *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_float_array(int size, float **data)
{
  initialize_numpy();		// required before using NumPy.

  int dimensions[1] = {size};
  PyObject *a = allocate_python_array(1, dimensions, NPY_FLOAT);
  if (data)
    *data = (float *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_float_array(int size1, int size2, float **data)
{
  initialize_numpy();		// required before using NumPy.

  int dimensions[2] = {size1, size2};
  PyObject *a = allocate_python_array(2, dimensions, NPY_FLOAT);
  if (data)
    *data = (float *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
PyObject *python_float_array(int size1, int size2, int size3, float **data)
{
  initialize_numpy();		// required before using NumPy.

  int dimensions[3] = {size1, size2, size3};
  PyObject *a = allocate_python_array(3, dimensions, NPY_FLOAT);
  if (data)
    *data = (float *)PyArray_DATA((PyArrayObject *)a);

  return a;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_3d_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  try
    {
      *na = array_from_python(arg, 3);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_3d_array(PyObject *arg, void *array)
{
  Numeric_Array *na = static_cast<Numeric_Array *>(array);
  try
    {
      *na = array_from_python(arg, 3, false);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }
  return 1;
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
  try
    {
      python_array_to_c(arg, static_cast<int*>(i3), 3);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_3_array(PyObject *arg, void *f3)
{
  try
    {
      python_array_to_c(arg, static_cast<float*>(f3), 3);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_double_3_array(PyObject *arg, void *f3)
{
  try
    {
      python_array_to_c(arg, static_cast<double*>(f3), 3);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }
  return 1;
}

// ----------------------------------------------------------------------------
//
static int parse_float_n(PyObject *arg, void *farray, bool allow_copy)
{
  try
    {
      Numeric_Array v = array_from_python(arg, 1, Numeric_Array::Float, allow_copy);
      *static_cast<FArray*>(farray) = static_cast<FArray>(v);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_n_array(PyObject *arg, void *farray)
  { return parse_float_n(arg, farray, true); }
extern "C" int parse_writable_float_n_array(PyObject *arg, void *farray)
  { return parse_float_n(arg, farray, false); }

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_3x4_array(PyObject *arg, void *f34)
{
  try
    {
      python_array_to_c(arg, static_cast<float*>(f34), 3, 4);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_double_3x4_array(PyObject *arg, void *d34)
{
  try
    {
      python_array_to_c(arg, static_cast<double*>(d34), 3, 4);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }
  return 1;
}

// ----------------------------------------------------------------------------
//
static int parse_float_n3(PyObject *arg, void *farray, bool allow_copy, bool f64)
{
  try
    {
      Numeric_Array::Value_Type ftype = (f64 ? Numeric_Array::Double : Numeric_Array::Float);
      Numeric_Array v = array_from_python(arg, 0, ftype, allow_copy);
      if (v.dimension() == 1 && v.size() == 0)
	{
	  int size[2] = {0,3};
	  v = Numeric_Array(ftype, 2, size);
	}
      if (v.dimension() != 2)
	{
	  std::ostringstream msg;
	  msg << "Array must be 2 dimensional, got "
	      << v.dimension() << " dimensional";
	  throw std::runtime_error(msg.str());
	}
      if (v.size(1) != 3)
	throw std::runtime_error("Second array dimension must have size 3.");
      if (f64)
	*static_cast<DArray*>(farray) = static_cast<DArray>(v);
      else
	*static_cast<FArray*>(farray) = static_cast<FArray>(v);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }

  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_n3_array(PyObject *arg, void *farray)
  { return parse_float_n3(arg, farray, true, false); }
extern "C" int parse_writable_float_n3_array(PyObject *arg, void *farray)
  { return parse_float_n3(arg, farray, false, false); }
extern "C" int parse_double_n3_array(PyObject *arg, void *darray)
  { return parse_float_n3(arg, darray, true, true); }
extern "C" int parse_writable_double_n3_array(PyObject *arg, void *darray)
  { return parse_float_n3(arg, darray, false, true); }

// ----------------------------------------------------------------------------
//
extern "C" int parse_writable_float_3d_array(PyObject *arg, void *farray)
{
  try
    {
      Numeric_Array v = array_from_python(arg, 3, Numeric_Array::Float, false);
      *static_cast<FArray*>(farray) = static_cast<FArray>(v);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }

  return 1;
}

// ----------------------------------------------------------------------------
//
static int parse_int_nm(PyObject *arg, int m, void *iarray, bool allow_copy)
{
  try
    {
      Numeric_Array v = array_from_python(arg, 0, allow_copy);
      if (v.dimension() == 1 && v.size() == 0)
	{
	  int size[2] = {0,3};
	  v = Numeric_Array(Numeric_Array::Int, 2, size);
	}
      if (v.dimension() != 2)
	{
	  std::ostringstream msg;
	  msg << "Array must be 2 dimensional, got "
	      << v.dimension() << " dimensional";
	  throw std::runtime_error(msg.str());
	}
      if (v.value_type() == Numeric_Array::Long_Int && allow_copy)
	{
	  IArray vi = IArray(v.dimension(), v.sizes());
	  vi.set(Reference_Counted_Array::Array<long int>(v));
	  v = Numeric_Array(Numeric_Array::Int, vi);
	}
      if (v.value_type() != Numeric_Array::Int)
	{
	  std::ostringstream msg;
	  msg << "array type must be int or long int, got "
	      << Numeric_Array::value_type_name(v.value_type());
	  throw std::runtime_error(msg.str());
	}
      if (v.size(1) != m)
	{
	  std::ostringstream msg;
	  msg << "Second array dimension must have size " << m
	      << ", got " << v.size(1) << std::endl;
	  throw std::runtime_error(msg.str());
	}
      *static_cast<IArray*>(iarray) = static_cast<IArray>(v);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }

  return 1;
}

// ----------------------------------------------------------------------------
//
static int parse_int_array(PyObject *arg, void *iarray, bool allow_copy)
{
  try
    {
      Numeric_Array v = array_from_python(arg, 1, Numeric_Array::Int,
					  allow_copy);
      *static_cast<IArray*>(iarray) = static_cast<IArray>(v);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return 0;
    }

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

  int *sizes = new int[dim+1];
  for (int k = 0 ; k < dim ; ++k)
    sizes[k] = PyArray_DIM(a,k);
  sizes[dim] = PyArray_ITEMSIZE(a);

  //
  // NumPy strides are in bytes.
  // Numeric_Array strides are in elements.
  //
  long *strides = new long[dim+1];
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
// Need to call NumPy import_array() before using NumPy routines.
//
static void *initialize_numpy()
{
  static bool first_call = true;
  if (first_call)
    {
      first_call = false;
      import_array();
    }
  return NULL;
}
