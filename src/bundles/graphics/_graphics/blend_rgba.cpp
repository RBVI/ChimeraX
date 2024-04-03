// vi: set expandtab shiftwidth=4 softtabstop=4:

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
// Blend images for motion blur.
//
#include <Python.h>			// use PyObject
#include <math.h>			// use ceil, floor

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use Numeric_Array, Array<T>

// ----------------------------------------------------------------------------
//
template<class T>
static void blend_colors(float f, const Reference_Counted_Array::Array<T> &m1,
			 const Reference_Counted_Array::Array<T> &m2,
			 const Reference_Counted_Array::Array<T> &bgcolor,
			 float alpha,
			 const Reference_Counted_Array::Array<T> &m,
			 int64_t *count)
			   
{
  int64_t n = m.size(), c = 0;
  T *v1 = m1.values(), *v2 = m2.values(), *v = m.values(), *bg = bgcolor.values();
  T bg0 = bg[0], bg1 = bg[1], bg2 = bg[2], a = static_cast<T>(floor(alpha));;
  for (int k = 0 ; k < n ; k += 4)
    {
      if (v1[k] != bg0 || v1[k+1] != bg1 || v1[k+2] != bg2)
	{ v[k] = v1[k]; v[k+1] = v1[k+1]; v[k+2] = v1[k+2]; }
      else
	{
	  float f0 = f*(static_cast<float>(v2[k])-bg0);
	  float f1 = f*(static_cast<float>(v2[k+1])-bg1);
	  float f2 = f*(static_cast<float>(v2[k+2])-bg2);
	  // Round integral types towards bgcolor.
	  v[k] = (f0 >= 0 ? static_cast<T>(floor(bg0+f0)) : static_cast<T>(ceil(f0+bg0)));
	  v[k+1] = (f1 >= 0 ? static_cast<T>(floor(bg1+f1)) : static_cast<T>(ceil(f1+bg1)));
	  v[k+2] = (f2 >= 0 ? static_cast<T>(floor(bg2+f2)) : static_cast<T>(ceil(f2+bg2)));
	  if (f0 != 0 || f1 != 0 || f2 != 0)
	    c += 1;
	}
      v[k+3] = a;
    }
  *count = c;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *blur_blend_images(PyObject *, PyObject *args, PyObject *keywds)
{
  Reference_Counted_Array::Numeric_Array m1, m2, m, bgcolor;
  PyObject *bgcolor_py;
  float f, alpha;
  const char *kwlist[] = {"f", "rgba1", "rgba2", "bgcolor", "alpha", "rgba", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("fO&O&OfO&"), (char **)kwlist,
				   &f, parse_3d_array, &m1, parse_3d_array, &m2,
				   &bgcolor_py, &alpha, parse_3d_array, &m))
    return NULL;

  if (!m1.is_contiguous() || !m2.is_contiguous() || !m.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "blend_images: arrays must be contiguous");
      return NULL;
    }
  if (m1.value_type() != m.value_type() || m2.value_type() != m.value_type())
    {
      PyErr_SetString(PyExc_TypeError, "blend_images: arrays must have same value type");
      return NULL;
    }
  if (m1.size() != m.size() || m2.size() != m.size())
    {
      PyErr_SetString(PyExc_TypeError, "blend_images: arrays must have same size");
      return NULL;
    }
  if (m1.size(2) != 4 || m2.size(2) != 4 || m.size(2) != 4)
    {
      PyErr_SetString(PyExc_TypeError, "blend_images: arrays must have third dimension of size 4");
      return NULL;
    }
  if (!array_from_python(bgcolor_py, 1, m.value_type(), &bgcolor))
    return NULL;
  if (bgcolor.size() != 3 || !bgcolor.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "blend_images: bgcolor must be contiguous 3 element array");
      return NULL;
    }
  int64_t count;
  call_template_function(blend_colors, m.value_type(), (f, m1, m2, bgcolor, alpha, m, &count));

  PyObject *count_py = PyLong_FromLong(count);
  return count_py;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *accumulate_images(PyObject *, PyObject *args, PyObject *keywds)
{
  Reference_Counted_Array::Numeric_Array rgba8, rgba32;
  const char *kwlist[] = {"rgba8", "rgba32", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&"), (char **)kwlist,
				   parse_3d_array, &rgba8, parse_3d_array, &rgba32))
    return NULL;

  if (!rgba8.is_contiguous() || !rgba32.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "accumulate_images: arrays must be contiguous");
      return NULL;
    }
  if (rgba8.value_type() != Reference_Counted_Array::Numeric_Array::Unsigned_Char ||
      rgba32.value_type() !=  Reference_Counted_Array::Numeric_Array::Unsigned_Int)
    {
      PyErr_SetString(PyExc_TypeError, "accumulate_images: arrays must have value type uint8 and uint32");
      return NULL;
    }
  if (rgba8.size() != rgba32.size())
    {
      PyErr_SetString(PyExc_TypeError, "accumulate_images: arrays must have same size");
      return NULL;
    }

  unsigned char *v8 = static_cast<unsigned char *>(rgba8.values());
  unsigned int *v32 = static_cast<unsigned int *>(rgba32.values());
  int64_t n = rgba8.size();
  for (int64_t i = 0 ; i < n ; ++i)
    v32[i] += v8[i];

  return python_none();
}
