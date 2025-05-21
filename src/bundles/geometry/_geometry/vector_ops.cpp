// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
// Compute linear combination of matrices.  5x faster than numpy.
//
#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use Numeric_Array, Array<T>
#include "vector_ops.h"

// ----------------------------------------------------------------------------
//
template<class T>
static void inner(const Reference_Counted_Array::Array<T> &m1,
		  const Reference_Counted_Array::Array<T> &m2,
		  double *sum)
{
  double s = 0;
  int64_t n = m1.size(), s1 = m1.stride(0), s2 = m2.stride(0);
  T *v1 = m1.values(), *v2 = m2.values();
  if (s1 == 1 && s2 == 1)
    for (int64_t k = 0 ; k < n ; ++k)
      s += ((double)v1[k])*((double)v2[k]);
  else
    for (int64_t k = 0 ; k < n ; ++k)
      s += ((double)v1[k*s1])*((double)v2[k*s2]);
  *sum = s;
}

const char *inner_product_64_doc =
  "inner_product_64(u, v) -> float64\n"
  "\n"
  "Return the inner-product of two vectors accumulated as a 64-bit floating point value\n"
  "even if the vector components are 32-bit values.  Input arrays are 1-dimensional with\n"
  "any value type.\n";

// ----------------------------------------------------------------------------
// Computes sum in 64-bit, 1-d contiguous arrays only.
//
extern "C" PyObject *inner_product_64(PyObject *, PyObject *args, PyObject *keywds)
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
