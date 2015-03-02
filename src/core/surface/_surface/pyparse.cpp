// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject
#include <stdexcept>			// use std::runtime_error

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray, IArray

// ----------------------------------------------------------------------------
//
bool convert_vertex_array(PyObject *vertex_array, FArray *va,
			  bool allow_copy, bool contiguous)
{
  Numeric_Array na;
  if (!array_from_python(vertex_array, 2, Numeric_Array::Float, &na, allow_copy))
    return false;
  if (na.size(1) != 3)
    {
      PyErr_Format(PyExc_TypeError, "vertex array second dimension must be 3");
      return false;
    }
  *va = (contiguous ? na.contiguous_array() : na);
  return true;
}

// ----------------------------------------------------------------------------
//
bool convert_triangle_array(PyObject *triangle_array, IArray *ta,
			    bool allow_copy, bool contiguous)
{
  Numeric_Array na;
  if (!array_from_python(triangle_array, 2, Numeric_Array::Int, &na, allow_copy))
    return false;
  if (na.size(1) != 3)
    {
      PyErr_Format(PyExc_TypeError, "triangle array second dimension must be 3");
      return false;
    }
  *ta = (contiguous ? na.contiguous_array() : na);
  return true;
}
