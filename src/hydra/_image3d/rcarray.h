// ----------------------------------------------------------------------------
// Multi-dimensional reference counted arrays.
//
// This was written to avoid copying large numeric arrays, and to be able
// to easily apply calculations to arrays of all C language numeric types
// using template functions.
//
// Several array objects can point to the same data.  A reference count
// is maintained so when the last array object is delete, the data can
// be freed.
//
// Subarrays can be made which point to the same data.
//
// The Numeric_Array type can hold all C numeric data types.
// The call_template_function() macro encodes a switch statement that
// calls a template function with the value type matching the data.
//
// Both Numeric_Array and Array<T> have Untyped_Array as a base class.
// Assigning or contructing and Array<T> from a Numeric_Array does not
// cast the data values to type T.  The data is not copied or changed so
// it acts like a reinterpret_cast.  This can be very dangerous because
// it is easy to unintentionally convert a Numeric_Array to an Array<T>
// that does not match the type of the Numeric_Array.  For example a
// Numeric_Array of integers could be passed to a function taking an
// Array<float> without any compiler warnings.
//
#ifndef RCARRAY_HEADER_INCLUDED
#define RCARRAY_HEADER_INCLUDED

#include "refcount.h"		// use Reference_Count
#include <stdexcept>		// for std::invalid_argument in rcarrayt.cpp

namespace Reference_Counted_Array
{

// ----------------------------------------------------------------------------
// Used by Array.
//
class Release_Data
{
public:
  virtual ~Release_Data() {}
};

// ----------------------------------------------------------------------------
//
template<class T>
class Delete_Data : public Release_Data
{
public:
  Delete_Data(T *data) { this->data = data; }
  ~Delete_Data() { delete [] data; data = (T *)0; }
private:
  T *data;
};

// ----------------------------------------------------------------------------
// Multi-dimensional reference counted array with unspecified value type.
//
class Untyped_Array
{
public:
  Untyped_Array();
  Untyped_Array(int element_size, int dim, const int *size);
  Untyped_Array(int element_size, int dim, const int *size,
		const void *values);  // Copies values
  // The release object is destroyed when data values are no longer needed.
  Untyped_Array(int element_size, int dim, const int *size, void *values,
		Release_Data *release);
  Untyped_Array(int element_size, int dim, const int *sizes,
		const long *strides, void *data, Release_Data *release);
  Untyped_Array(const Untyped_Array &);
  virtual ~Untyped_Array();
  const Untyped_Array &operator=(const Untyped_Array &array);
  //  Untyped_Array copy() const;	// Copies values, not implemented

  int element_size() const;
  int dimension() const;
  int size(int axis) const;
  long size() const;
  const int *sizes() const;
  long stride(int axis) const;
  const long *strides() const;
  void *values() const;

  //
  // The slice method fixes the index for one of the dimensions giving an array
  // of one less dimension.  The data is not copied.  Modifying the slice data
  // values will effect the parent array.
  //
  Untyped_Array slice(int axis, int index) const;
  Untyped_Array subarray(int axis, int i_min, int i_max);
  bool is_contiguous() const;

  // Allows super classes to retrieve Python source data object for array.
  Release_Data *release_method() const { return release_data; }

private:
  void *data;
  Reference_Count data_reference_count;
  Release_Data *release_data;
  int element_siz;
  int dim;
  long start;
  long *stride_size;
  int *siz;

  void initialize(int element_size, int dim, const int *size, bool allocate);
};

template <class T> class Array_Operator;

// ----------------------------------------------------------------------------
// Multi-dimensional reference counted array with specified value type T.
//
template <class T>
class Array : public Untyped_Array
{
public:
  Array();
  Array(int dim, const int *size);
  Array(int dim, const int *size, const T *values);	// copies values
  // The release object is destroyed when data values are no longer needed.
  Array(int dim, const int *size, T *values, Release_Data *release);
  Array(const Untyped_Array &);
  Array(const Array<T> &);
  virtual ~Array();
  const Array<T> &operator=(const Array<T> &array);
  Array<T> copy() const;

  T value(const int *index) const;
  T *values() const;
  T *copy_of_values() const;			// allocated with new []
  void get_values(T *v) const;
  void set(const int *index, T value);
  void set(T value);
  template <class S> void set(const Array<S> &a);  // cast values if needed
  void apply(class Array_Operator<T> &op);
  //
  // The slice method fixes the index for one of the dimensions giving an array
  // of one less dimension.  The data is not copied.  Modifying the slice data
  // values will effect the parent array.
  //
  Array<T> slice(int axis, int index) const;
  Array<T> subarray(int axis, int i_min, int i_max);
  Array<T> contiguous_array() const;
};

// ----------------------------------------------------------------------------
// Multi-dimensional reference counted array with numeric value type.
//
class Numeric_Array : public Untyped_Array
{
 public:
  enum Value_Type { Char, Signed_Char, Unsigned_Char,
		    Short_Int, Unsigned_Short_Int, Int, Unsigned_Int,
		    Long_Int, Unsigned_Long_Int, Float, Double };
  static int size_of_type(Value_Type t);
  static const char *value_type_name(Value_Type type);

  Numeric_Array();
  Numeric_Array(Value_Type type, int dim, const int *sizes);
  Numeric_Array(Value_Type type, int dim, const int *sizes, const long *strides,
		void *data, Release_Data *release);
  Numeric_Array(const Numeric_Array &);
  Numeric_Array(Value_Type type, const Untyped_Array &);
  Numeric_Array &operator=(const Numeric_Array &);
  Numeric_Array copy() const;
  Numeric_Array as_type(Value_Type type);

  Value_Type value_type() const;

  Numeric_Array slice(int axis, int index) const;
  Numeric_Array contiguous_array() const;

 private:
  Value_Type type;
};

// ----------------------------------------------------------------------------
// f should be a template function taking an Array<T> argument.
// args should contain a Numeric_Array argument in the corresponding
// position which will be cast to the correct Array<T> type.
//
#define RCANA Reference_Counted_Array::Numeric_Array
#define call_template_function(f, value_type, args) \
  switch (value_type) \
    { \
    case RCANA::Char: f<char> args; break; \
    case RCANA::Signed_Char: f<signed char> args; break; \
    case RCANA::Unsigned_Char: f<unsigned char> args; break; \
    case RCANA::Short_Int: f<short> args; break; \
    case RCANA::Unsigned_Short_Int: f<unsigned short> args; break; \
    case RCANA::Int: f<int> args; break; \
    case RCANA::Unsigned_Int: f<unsigned int> args; break; \
    case RCANA::Long_Int: f<long> args; break; \
    case RCANA::Unsigned_Long_Int: f<unsigned long> args; break; \
    case RCANA::Float: f<float> args; break; \
    case RCANA::Double: f<double> args; break; \
    }

// ----------------------------------------------------------------------------
//
template <class T>
class Array_Operator
{
 public:
  virtual ~Array_Operator() {}
  virtual T operator()(T x) = 0;
};

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array() : Untyped_Array()
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(int dim, const int *size) : Untyped_Array(sizeof(T), dim, size)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(int dim, const int *size, const T *values) :
  Untyped_Array(sizeof(T), dim, size, (void *)values)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(int dim, const int *size, T *values, Release_Data *release) :
  Untyped_Array(sizeof(T), dim, size, (void *)values, release)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(const Untyped_Array &a) : Untyped_Array(a)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(const Array<T> &a) : Untyped_Array(a)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
const Array<T> &Array<T>::operator=(const Array<T> &array)
{
  Untyped_Array::operator=(array);
  return *this;
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::~Array()
{
}

// ----------------------------------------------------------------------------
//
template <class T>
T Array<T>::value(const int *index) const
{
  long i = 0;
  int dim = dimension();
  for (int a = 0 ; a < dim ; ++a)
    i += index[a] * stride(a);
  T *data = values();
  return data[i];
}

// ----------------------------------------------------------------------------
//
template <class T>
T *Array<T>::values() const
{
  return (T *) Untyped_Array::values();
}

// ----------------------------------------------------------------------------
//
template <class T>
T *Array<T>::copy_of_values() const
{
  long s = size();
  T *v = new T[s];
  get_values(v);
  return v;
}

// ----------------------------------------------------------------------------
// Optimized for dimension <= 4.
//
template <class T>
void Array<T>::get_values(T *v) const
{
  if (is_contiguous())
    {
      T *d = values();
      long length = size();
      for (long i = 0 ; i < length ; ++i)
	v[i] = d[i];
      return;
    }

  if (dimension() == 0)
    return;

  T *d = values();

  long k = 0;
  long i0, j0, js0 = stride(0), s0 = size(0);
  if (dimension() == 1)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	v[k++] = d[j0];
      return;
    }

  long i1, j1, js1 = stride(1), s1 = size(1);
  if (dimension() == 2)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  v[k++] = d[j1];
      return;
    }

  long i2, j2, js2 = stride(2), s2 = size(2);
  if (dimension() == 3)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    v[k++] = d[j2];
      return;
    }

  long i3, j3, js3 = stride(3), s3 = size(3);
  if (dimension() == 4)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    for (i3=0, j3=j2 ; i3<s3 ; ++i3, j3+=js3)
	      v[k++] = d[j3];
      return;
    }

  long step = size()/size(0);
  for (long i = 0 ; i < s0 ; ++i)
    slice(0,i).get_values(v + i*step);
}

// ----------------------------------------------------------------------------
//
template <class T>
void Array<T>::set(const int *index, T value)
{
  long i = 0;
  int dim = dimension();
  for (int a = 0 ; a < dim ; ++a)
    i += index[a] * stride(a);
  T *data = values();
  data[i] = value;
}

// ----------------------------------------------------------------------------
// Optimized for contiguous arrays and dimension <= 4.
//
template <class T>
void Array<T>::set(T value)
{
  if (is_contiguous())
    {
      T *d = values();
      long length = size();
      for (int i = 0 ; i < length ; ++i)
	d[i] = value;
      return;
    }

  if (dimension() == 0)
    return;

  T *d = values();

  long i0, j0, js0 = stride(0), s0 = size(0);
  if (dimension() == 1)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	d[j0] = value;
      return;
    }

  long i1, j1, js1 = stride(1), s1 = size(1);
  if (dimension() == 2)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  d[j1] = value;
      return;
    }

  long i2, j2, js2 = stride(2), s2 = size(2);
  if (dimension() == 3)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    d[j2] = value;
      return;
    }

  long i3, j3, js3 = stride(3), s3 = size(3);
  if (dimension() == 4)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    for (i3=0, j3=j2 ; i3<s3 ; ++i3, j3+=js3)
	      d[j3] = value;
      return;
    }

  for (long i = 0 ; i < s0 ; ++i)
    this->slice(0,i).set(value);
}

// ----------------------------------------------------------------------------
// Optimized for dimension <= 4.
//
template <class T>
template <class S>
void Array<T>::set(const Array<S> &a)
{
  if (a.dimension() != dimension())
    throw std::invalid_argument("void Array::set(): dimension mismatch");

  if (dimension() == 0)
    return;

  T *d = values();
  S *ad = a.values();
  
  long i0, j0, js0 = stride(0), k0, ks0 = a.stride(0);
  int s0 = (size(0) < a.size(0) ? size(0) : a.size(0));
  if (dimension() == 1)
    {
      for (i0=0, j0=0, k0=0 ; i0<s0 ; ++i0, j0+=js0, k0+=ks0)
	d[j0] = static_cast<T>(ad[k0]);
      return;
    }


  long i1, j1, js1 = stride(1), k1, ks1 = a.stride(1);
  int s1 = (size(1) < a.size(1) ? size(1) : a.size(1));
  if (dimension() == 2)
    {
      for (i0=0, j0=0, k0=0 ; i0<s0 ; ++i0, j0+=js0, k0+=ks0)
	for (i1=0, j1=j0, k1=k0 ; i1<s1 ; ++i1, j1+=js1, k1+=ks1)
	  d[j1] = static_cast<T>(ad[k1]);
      return;
    }

  long i2, j2, js2 = stride(2), k2, ks2 = a.stride(2);
  int s2 = (size(2) < a.size(2) ? size(2) : a.size(2));
  if (dimension() == 3)
    {
      for (i0=0, j0=0, k0=0 ; i0<s0 ; ++i0, j0+=js0, k0+=ks0)
	for (i1=0, j1=j0, k1=k0 ; i1<s1 ; ++i1, j1+=js1, k1+=ks1)
	  for (i2=0, j2=j1, k2=k1 ; i2<s2 ; ++i2, j2+=js2, k2+=ks2)
	    d[j2] = static_cast<T>(ad[k2]);
      return;
    }

  long i3, j3, js3 = stride(3), k3, ks3 = a.stride(3);
  int s3 = (size(3) < a.size(3) ? size(3) : a.size(3));
  if (dimension() == 4)
    {
      for (i0=0, j0=0, k0=0 ; i0<s0 ; ++i0, j0+=js0, k0+=ks0)
	for (i1=0, j1=j0, k1=k0 ; i1<s1 ; ++i1, j1+=js1, k1+=ks1)
	  for (i2=0, j2=j1, k2=k1 ; i2<s2 ; ++i2, j2+=js2, k2+=ks2)
	    for (i3=0, j3=j2, k3=k2 ; i3<s3 ; ++i3, j3+=js3, k3+=ks3)
	      d[j3] = static_cast<T>(ad[k3]);
      return;
    }

  for (long i = 0 ; i < s0 ; ++i)
    this->slice(0,i).set(a.slice(0,i));
}

// ----------------------------------------------------------------------------
// Optimized for dimension <= 4.
//
template <class T>
void Array<T>::apply(Array_Operator<T> &op)
{
  if (dimension() == 0)
    return;

  T *d = values();
  
  long i0, j0, js0 = stride(0), s0 = size(0);
  if (dimension() == 1)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	d[j0] = op(d[j0]);
      return;
    }

  long i1, j1, js1 = stride(1), s1 = size(1);
  if (dimension() == 2)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  d[j1] = op(d[j1]);
      return;
    }

  long i2, j2, js2 = stride(2), s2 = size(2);
  if (dimension() == 3)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    d[j2] = op(d[j2]);
      return;
    }

  long i3, j3, js3 = stride(3), s3 = size(3);
  if (dimension() == 4)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    for (i3=0, j3=j2 ; i3<s3 ; ++i3, j3+=js3)
	      d[j3] = op(d[j3]);
      return;
    }

  for (long i = 0 ; i < s0 ; ++i)
    this->slice(0,i).apply(op);
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T> Array<T>::slice(int axis, int index) const
{
  return Array<T>(Untyped_Array::slice(axis, index));
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T> Array<T>::subarray(int axis, int i_min, int i_max)
{
  return Array<T>(Untyped_Array::subarray(axis, i_min, i_max));
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T> Array<T>::contiguous_array() const
{
  if (is_contiguous())
    return *this;

  return this->copy();
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T> Array<T>::copy() const
{
  T *v = copy_of_values();
  return Array<T>(dimension(), sizes(), v, new Delete_Data<T>(v));
}

}  // end of namespace Reference_Counted_Array

typedef Reference_Counted_Array::Array<float> FArray;
typedef Reference_Counted_Array::Array<double> DArray;
typedef Reference_Counted_Array::Array<int> IArray;
typedef Reference_Counted_Array::Array<char> CArray;

#endif
