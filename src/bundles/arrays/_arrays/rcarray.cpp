// vi: set expandtab ts=4 sw=4:

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
//
#include <string.h>     // use memcpy()
#include <sstream>	// use std::ostringstream

#define ARRAYS_EXPORT
#include "rcarray.h"        // use Untyped_Array, Numeric_Array

namespace Reference_Counted_Array
{

// ----------------------------------------------------------------------------
//
Untyped_Array::Untyped_Array()
{
  initialize(0, 0, (int64_t *) 0, false);
}

// ----------------------------------------------------------------------------
//
Untyped_Array::Untyped_Array(int element_size, int dim, const int64_t *size)
{
  initialize(element_size, dim, size, true);
}

// ----------------------------------------------------------------------------
//
void Untyped_Array::initialize(int element_size, int dim, const int64_t *size,
                   bool allocate)
{
  if (allocate && dim > 0)
    {
      int64_t length = 1;
      for (int a = 0 ; a < dim ; ++a)
	length *= size[a];
      char *d = new char[length * element_size];
      this->data = (void *) d;
      this->release_data = new Delete_Data<char>(d);
    }
  else
    {
      this->data = (void *) 0;
      this->release_data = (Release_Data *) 0;
    }

  this->start = 0;
  this->element_siz = element_size;
  this->dim = dim;

  this->siz = (dim > 0 ? new int64_t[dim] : (int64_t *) 0);
  for (int a = 0 ; a < dim ; ++a)
    this->siz[a] = size[a];

  this->stride_size = (dim > 0 ? new int64_t[dim] : (int64_t *) 0);
  int64_t stride = 1;
  for (int a = dim-1 ; a >= 0 ; stride *= size[a], --a)
    this->stride_size[a] = stride;
}

// ----------------------------------------------------------------------------
//
Untyped_Array::Untyped_Array(int element_size, int dim, const int64_t *size,
                 const void *values)
{
  initialize(element_size, dim, size, true);

  int64_t length = this->size() * element_size;
  memcpy(data, values, length);
}

// ----------------------------------------------------------------------------
//
Untyped_Array::Untyped_Array(int element_size, int dim, const int64_t *size,
                 void *values, Release_Data *release)
{
  initialize(element_size, dim, size, false);

  this->data = values;
  this->release_data = release;
}

// ----------------------------------------------------------------------------
//
Untyped_Array::Untyped_Array(int element_size, int dim,
                 const int64_t *sizes, const int64_t *strides,
                 void *data, Release_Data *release)
{
  initialize(element_size, dim, sizes, false);

  for (int a = 0 ; a < dim ; ++a)
    this->stride_size[a] = strides[a];

  this->data = data;
  this->release_data = release;
}

// ----------------------------------------------------------------------------
//
Untyped_Array::Untyped_Array(const Untyped_Array &a)
{
  initialize(0, 0, (int64_t *) 0, false);
  *this = a;
}

// ----------------------------------------------------------------------------
//
const Untyped_Array &Untyped_Array::operator=(const Untyped_Array &array)
{
  if (&array == this)
    return *this;   // Avoid deleting data on self assignment.
  
  if (data_reference_count.reference_count() == 1)
    delete release_data;
  delete [] stride_size;
  delete [] siz;

  this->data = array.data;
  this->data_reference_count = array.data_reference_count;
  this->release_data = array.release_data;
  this->start = array.start;
  this->element_siz = array.element_size();
  this->dim = array.dim;
  this->stride_size = new int64_t[dim];
  this->siz = new int64_t[dim];
  for (int a = 0 ; a < dim ; ++a)
    {
      this->siz[a] = array.siz[a];
      this->stride_size[a] = array.stride_size[a];
    }
  return *this;
}

// ----------------------------------------------------------------------------
//
Untyped_Array::~Untyped_Array()
{
  if (data_reference_count.reference_count() == 1)
    delete release_data;
  data = (void *)0;
  release_data = (Release_Data *)0;
  delete [] stride_size;
  stride_size = (int64_t *)0;
  delete [] siz;
  siz = (int64_t *)0;
}

// ----------------------------------------------------------------------------
// TODO: Does not copy values as it is supposed to.
//
/*
Untyped_Array Untyped_Array::copy() const
{
  return Untyped_Array(element_size(), dimension(), sizes(), values())
}
*/

// ----------------------------------------------------------------------------
//
int Untyped_Array::element_size() const
{
  return element_siz;
}

// ----------------------------------------------------------------------------
//
int Untyped_Array::dimension() const
{
  return dim;
}

// ----------------------------------------------------------------------------
//
int64_t Untyped_Array::size(int axis) const
{
  return siz[axis];
}

// ----------------------------------------------------------------------------
//
int64_t Untyped_Array::size() const
{
  if (dim == 0)
    return 0;

  int64_t s = 1;
  for (int a = 0 ; a < dim ; ++a)
    s *= size(a);

  return s;
}

// ----------------------------------------------------------------------------
//
const int64_t *Untyped_Array::sizes() const
{
  return siz;
}

// ----------------------------------------------------------------------------
//
std::string Untyped_Array::size_string() const
{
  std::ostringstream string;
  for (int a = 0 ; a < dim ; ++a)
    {
      string << siz[a];
      if (a+1 < dim)
	string << ", ";
    }
  return string.str();
}

// ----------------------------------------------------------------------------
//
std::string Untyped_Array::size_string(int axis) const
{
  std::ostringstream string;
  string << siz[axis];
  return string.str();
}

// ----------------------------------------------------------------------------
//
int64_t Untyped_Array::stride(int axis) const
{
  return stride_size[axis];
}

// ----------------------------------------------------------------------------
//
const int64_t *Untyped_Array::strides() const
{
  return stride_size;
}

// ----------------------------------------------------------------------------
//
void *Untyped_Array::values() const
{
  return (void *)(((char *)data) + start * element_siz);
}

// ----------------------------------------------------------------------------
//
Untyped_Array Untyped_Array::slice(int axis, int64_t index) const
{
  Untyped_Array s(*this);
  s.start = s.start + index * stride_size[axis];
  for (int a = axis ; a < dim-1 ; ++a)
    {
      s.siz[a] = s.siz[a+1];
      s.stride_size[a] = s.stride_size[a+1];
    }
  s.dim = s.dim - 1;
  return s;
}

// ----------------------------------------------------------------------------
//
Untyped_Array Untyped_Array::subarray(int axis, int64_t i_min, int64_t i_max)
{
  Untyped_Array s(*this);

  s.start += i_min*stride_size[axis];
  s.siz[axis] = i_max - i_min + 1;

  return s;
}

// ----------------------------------------------------------------------------
//
bool Untyped_Array::is_contiguous() const
{
  if (size() == 0)
    // Numpy 1.23 started using strides all 0 for 0-length arrays.  Bug #7384
    return true;
  int64_t contig_stride = 1;
  for (int a = dim-1 ; a >= 0 ; contig_stride *= size(a), --a)
    if (stride_size[a] != contig_stride)
      return false;
  return true;
}

// ----------------------------------------------------------------------------
//
int Numeric_Array::size_of_type(Value_Type t)
{
  switch (t)
    {
    case Char:          return sizeof(char);
    case Unsigned_Char:     return sizeof(unsigned char);
    case Signed_Char:       return sizeof(signed char);
    case Short_Int:     return sizeof(short int);
    case Unsigned_Short_Int:    return sizeof(unsigned short int);
    case Int:           return sizeof(int);
    case Unsigned_Int:      return sizeof(unsigned int);
    case Long_Int:      return sizeof(long int);
    case Unsigned_Long_Int: return sizeof(unsigned long int);
    case Float:         return sizeof(float);
    case Double:        return sizeof(double);
    };

  return 0;
}

// ----------------------------------------------------------------------------
//
Numeric_Array::Numeric_Array()
{
  this->type = Double;
}

// ----------------------------------------------------------------------------
//
Numeric_Array::Numeric_Array(Value_Type type, int dim, const int64_t *sizes) :
  Untyped_Array(size_of_type(type), dim, sizes)
{
  this->type = type;
}

// ----------------------------------------------------------------------------
//
Numeric_Array::Numeric_Array(Value_Type type, int dim,
                 const int64_t *sizes, const int64_t *strides,
                 void *data, Release_Data *release) :
  Untyped_Array(size_of_type(type), dim, sizes, strides, data, release)
{
  this->type = type;
}

// ----------------------------------------------------------------------------
//
Numeric_Array::Numeric_Array(const Numeric_Array &a) : Untyped_Array(a)
{
  this->type = a.value_type();
}

// ----------------------------------------------------------------------------
//
Numeric_Array::Numeric_Array(Value_Type type, const Untyped_Array &a) :
  Untyped_Array(a)
{
  this->type = type;
}

// ----------------------------------------------------------------------------
//
Numeric_Array &Numeric_Array::operator=(const Numeric_Array &a)
{
  Untyped_Array::operator=(a);
  this->type = a.value_type();
  return *this;
}

// ----------------------------------------------------------------------------
//
template <class T>
void cast_array(Numeric_Array &na, Numeric_Array::Value_Type type)
{
  Array<T> a(na.dimension(), na.sizes());
  call_template_function(a.template set, na.value_type(), (na));
  Numeric_Array nat(type, a);
  na = nat;
}

// ----------------------------------------------------------------------------
//
Numeric_Array Numeric_Array::as_type(Value_Type type)
{
  Numeric_Array ta = *this;
  if (type != value_type())
    call_template_function(cast_array, type, (ta, type));
  return ta;
}

// ----------------------------------------------------------------------------
//
Numeric_Array::Value_Type Numeric_Array::value_type() const
{
  return type;
}

// ----------------------------------------------------------------------------
//
Numeric_Array Numeric_Array::slice(int axis, int64_t index) const
{
  return Numeric_Array(value_type(), Untyped_Array::slice(axis, index));
}

// ----------------------------------------------------------------------------
//
template <class T>
static void make_copy(const Array<T> &a, Numeric_Array::Value_Type vtype,
                Numeric_Array *c)
{
  *c = Numeric_Array(vtype, a.copy());
}

// ----------------------------------------------------------------------------
//
Numeric_Array Numeric_Array::copy() const
{
  Numeric_Array c;
  call_template_function(make_copy, value_type(), (*this, value_type(), &c));
  return c;
}

// ----------------------------------------------------------------------------
//
template <class T>
static void make_contiguous(const Array<T> &a, Numeric_Array::Value_Type vtype,
                Numeric_Array *c)
{
  *c = Numeric_Array(vtype, a.contiguous_array());
}

// ----------------------------------------------------------------------------
//
Numeric_Array Numeric_Array::contiguous_array() const
{
  Numeric_Array c;
  call_template_function(make_contiguous, value_type(),
             (*this, value_type(), &c));
  return c;
}

// ----------------------------------------------------------------------------
//
const char *Numeric_Array::value_type_name(Value_Type type)
{
  switch (type)
    {
    case Char: return "char";
    case Signed_Char: return "signed char";
    case Unsigned_Char: return "unsigned char";
    case Short_Int: return "short int";
    case Unsigned_Short_Int: return "unsigned short int";
    case Int: return "int";
    case Unsigned_Int: return "unsigned int";
    case Long_Int: return "long int";
    case Unsigned_Long_Int: return "unsigned long int";
    case Float: return "float";
    case Double: return "double";
    }
  return "unknown";
}

}  // end of namespace Reference_Counted_Array
