// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Compute linear combination of matrices.  5x faster than numpy.
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use Numeric_Array, Array<T>

// ----------------------------------------------------------------------------
//
template<class T>
static void inner(const Reference_Counted_Array::Array<T> &m1,
		  const Reference_Counted_Array::Array<T> &m2,
		  double *sum)
{
  double s = 0;
  int n = m1.size(), s1 = m1.stride(0), s2 = m2.stride(0);
  T *v1 = m1.values(), *v2 = m2.values();
  if (s1 == 1 && s2 == 1)
    for (int k = 0 ; k < n ; ++k)
      s += ((double)v1[k])*((double)v2[k]);
  else
    for (int k = 0 ; k < n ; ++k)
      s += ((double)v1[k*s1])*((double)v2[k*s2]);
  *sum = s;
}

// ----------------------------------------------------------------------------
// Computes sum in 64-bit, 1-d contiguous arrays only.
//
extern "C" PyObject *inner_product_64(PyObject *s, PyObject *args, PyObject *keywds)
{
  Reference_Counted_Array::Numeric_Array m1, m2;
  const char *kwlist[] = {"m1", "m2", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&"), (char **)kwlist,
				   parse_1d_array, &m1,
				   parse_1d_array, &m2))
    return NULL;

  if (m2.size() != m1.size())
    {
      PyErr_SetString(PyExc_TypeError,
		      "inner_product_64: arrays must be same size");
      return NULL;
    }
  if (m2.value_type() != m1.value_type())
    {
      PyErr_SetString(PyExc_TypeError,
		      "inner_product_64: arrays must have same value type");
      return NULL;
    }

  double sum = 0;
  call_template_function(inner, m1.value_type(), (m1, m2, &sum));

  return PyFloat_FromDouble(sum);
}
