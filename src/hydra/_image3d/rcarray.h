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
#include "volumearray_config.h"	// use VOLUMEARRAY_IMEX

namespace Reference_Counted_Array
{

// ----------------------------------------------------------------------------
// Used by Array.
//
class VOLUMEARRAY_IMEX Release_Data
{
public:
  virtual ~Release_Data() {}
};

// ----------------------------------------------------------------------------
//
template<class T>
class VOLUMEARRAY_IMEX Delete_Data : public Release_Data
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
class VOLUMEARRAY_IMEX Untyped_Array
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
class VOLUMEARRAY_IMEX Array : public Untyped_Array
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
class VOLUMEARRAY_IMEX Numeric_Array : public Untyped_Array
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

#include "rcarrayt.cpp"		// Array template implementation

}  // end of namespace Reference_Counted_Array

typedef Reference_Counted_Array::Array<float> FArray;
typedef Reference_Counted_Array::Array<int> IArray;

#endif
