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

// ----------------------------------------------------------------------------
//
template<class T>
static void lin_combine(float f1, const Reference_Counted_Array::Array<T> &m1,
			float f2, const Reference_Counted_Array::Array<T> &m2,
			const Reference_Counted_Array::Array<T> &m)
			   
{
  int64_t n = m.size();
  T *v1 = m1.values(), *v2 = m2.values(), *v = m.values();
  for (int64_t k = 0 ; k < n ; ++k)
	 v[k] = static_cast<T>(f1*v1[k]+f2*v2[k]);
}

// ----------------------------------------------------------------------------
// Return linear combination of 3-d arrays.
//
extern "C" PyObject *linear_combination(PyObject *, PyObject *args, PyObject *keywds)
{
  Reference_Counted_Array::Numeric_Array m1, m2, m;
  float f1, f2;
  const char *kwlist[] = {"f1", "m1", "f2", "m2", "result", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("fO&fO&O&"), (char **)kwlist,
				   &f1, parse_3d_array, &m1, &f2, parse_3d_array, &m2,
				   parse_3d_array, &m))
    return NULL;

  if (!m1.is_contiguous() || !m2.is_contiguous() || !m.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "linear_combination: arrays must be contiguous");
      return NULL;
    }
  if (m1.value_type() != m.value_type() || m2.value_type() != m.value_type())
    {
      PyErr_SetString(PyExc_TypeError,
		      "linear_combination: arrays must have same value type");
      return NULL;
    }

  call_template_function(lin_combine, m.value_type(), (f1, m1, f2, m2, m));

  return python_none();
}
