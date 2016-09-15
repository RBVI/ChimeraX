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
// Compute linear combination of matrices.  5x faster than numpy.
//
#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use Numeric_Array, Array<T>

namespace Map_Cpp
{
// ----------------------------------------------------------------------------
//
template<class T>
static void local_corr(const Reference_Counted_Array::Array<T> &m1,
		       const Reference_Counted_Array::Array<T> &m2,
		       int window_size, bool subtract_mean, FArray &mc)
{
  int w = window_size;
  int n = w*w*w;
  int s0 = m1.size(0), s1 = m1.size(1), s2 = m1.size(2);
  int st0 = m1.stride(0), st1 = m1.stride(1), st2 = m1.stride(2);
  T *v1 = m1.values(), *v2 = m2.values();
  float *vc = mc.values();
  int cs0 = mc.stride(0), cs1 = mc.stride(1), cs2 = mc.stride(2);
  for (int i0 = 0 ; i0 < s0-w+1 ; ++i0)
    for (int i1 = 0 ; i1 < s1-w+1 ; ++i1)
      for (int i2 = 0 ; i2 < s2-w+1 ; ++i2)
	{
	  int i = i2*st2 + i1*st1 + i0*st0;
	  double v1v1 = 0, v2v2 = 0, v1v2 = 0, v1sum = 0, v2sum = 0;
	  for (int o0 = 0 ; o0 < w ; ++o0)
	    for (int o1 = 0 ; o1 < w ; ++o1) {
	      int j = i + o1*st1 + o0*st0;
	      for (int o2 = 0 ; o2 < w ; ++o2, j += st2) {
		double v1j = static_cast<double> (v1[j]);
		double v2j = static_cast<double> (v2[j]);
		v1sum += v1j;
		v2sum += v2j;
		v1v1 += v1j*v1j;
		v2v2 += v2j*v2j;
		v1v2 += v1j*v2j;
	      }
	    }
	  double vn, vip;
	  if (subtract_mean)
	    {
	      double vn2 = (v1v1 - v1sum*v1sum/n)*(v2v2 - v2sum*v2sum/n);
	      vn = (vn2 >= 0 ? sqrt(vn2) : 0);
	      vip = v1v2 - v1sum*v2sum/n;
	    }
	  else
	    {
	      vn = sqrt(v1v1*v2v2);
	      vip = v1v2;
	    }
	  int k = i2*cs2 + i1*cs1 + i0*cs0;
	  vc[k] = (vn > 0 ? vip / vn : 0);
	}
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
local_correlation(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array map1, map2;
  int window_size, subtract_mean;
  FArray mapc;
  const char *kwlist[] = {"map1", "map2", "window_size", "subtract_mean", "mapc", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&iiO&"),
				   (char **)kwlist,
				   parse_3d_array, &map1,
				   parse_3d_array, &map2,
				   &window_size,
				   &subtract_mean,
				   parse_writable_float_3d_array, &mapc))
    return NULL;

  if (!map1.is_contiguous() || !map2.is_contiguous() || !mapc.is_contiguous())
    {
      const char *aname = (!map1.is_contiguous() ? "map1" : (!map2.is_contiguous() ? "map2" : "mapc"));
      PyErr_Format(PyExc_TypeError, "%s array must be contiguous", aname);
      return NULL;
    }
  if (map1.value_type() != map2.value_type())
    {
      PyErr_Format(PyExc_TypeError, "input arrays must have same value type");
      return NULL;
    }
  if (map1.size(0) != map2.size(0) || 
      map1.size(1) != map2.size(1) || 
      map1.size(2) != map2.size(2))
    {
      PyErr_Format(PyExc_TypeError, "input arrays must have same size, %d %d %d and %d %d %d",
		   map1.size(0), map1.size(1), map1.size(2), map2.size(0), map2.size(1), map2.size(2));
      return NULL;
    }
  if (window_size < 1)
    {
      PyErr_Format(PyExc_TypeError, "window size must be >= 1, got %d", window_size);
      return NULL;
    }
  if (window_size > map1.size(0) ||
      window_size > map1.size(1) ||
      window_size > map1.size(2))
    {
      PyErr_Format(PyExc_TypeError, "window size (%d) must be <= array size %d %d %d",
		   window_size, map1.size(0), map1.size(1), map1.size(2));
      return NULL;
    }
  if (mapc.size(0) != map1.size(0)-window_size+1 || 
      mapc.size(1) != map1.size(1)-window_size+1 || 
      mapc.size(2) != map1.size(2)-window_size+1)
    {
      PyErr_Format(PyExc_TypeError,
		   "output array (%d %d %d) must window_size-1 (%d) smaller than input arrays (%d %d %d)",
		   mapc.size(0), mapc.size(1), mapc.size(2), window_size-1, map1.size(0), map1.size(1), map1.size(2));
      return NULL;
    }

  call_template_function(local_corr, map1.value_type(),
			 (map1, map2, window_size, subtract_mean, mapc));
  return python_none();
}

}	// end of namespace Map_Cpp
