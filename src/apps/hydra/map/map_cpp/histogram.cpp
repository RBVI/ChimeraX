// ----------------------------------------------------------------------------
// Count values in bins for Numeric Python arrays.
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use Numeric_Array, Array<T>

namespace Map_Cpp
{

// ----------------------------------------------------------------------------
// Return minimum and maximum values of array elements.
//
template<class T>
static void min_and_max(const Reference_Counted_Array::Array<T> &seq,
			double *min, double *max)
{
  int n = seq.size();
  if (n == 0)
    { *min = *max = 0; return; }

  T *data = seq.values();
  double minimum = static_cast<double> (data[0]);
  double maximum = minimum;

  int dim = seq.dimension();
  int m0 = 1, m1 = 1, m2 = 1, s0 = 0, s1 = 0, s2 = 0;
  if (dim == 1)
    { m2 = seq.size(0); s2 = seq.stride(0); }
  else if (dim == 2)
    { s1 = seq.stride(0); s2 = seq.stride(1);
      m1 = seq.size(0); m2 = seq.size(1); }
  else if (dim == 3)
    { s0 = seq.stride(0); s1 = seq.stride(1); s2 = seq.stride(2);
      m0 = seq.size(0); m1 = seq.size(1); m2 = seq.size(2); }

  int i = 0;
  for (int i0 = 0 ; i0 < m0 ; ++i0, i += s0 - m1*s1)
    for (int i1 = 0 ; i1 < m1 ; ++i1, i += s1 - m2*s2)
      for (int i2 = 0 ; i2 < m2 ; ++i2, i += s2)
	{
	  double v = static_cast<double> (data[i]);
	  if (v > maximum)
	    maximum = v;
	  else if (v < minimum)
	    minimum = v;
	}

  *min = minimum;
  *max = maximum;
}

// ----------------------------------------------------------------------------
// Return minimum and maximum values of Numeric array elements.
//
extern "C" PyObject *
minimum_and_maximum(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array seq;
  const char *kwlist[] = {"array", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_array, &seq))
    return NULL;

  if (seq.dimension() < 0 || seq.dimension() > 3)
    {
      PyErr_SetString(PyExc_TypeError,
		      "minimum_and_maximum(): array must be 1, 2, or 3 dimensional");
      return NULL;
    }

  double min, max;
  call_template_function(min_and_max, seq.value_type(), (seq, &min, &max));

  return python_tuple(PyFloat_FromDouble(min), PyFloat_FromDouble(max));
}

// ----------------------------------------------------------------------------
// Return a Numeric array of integers that counts the number of data values
// in a series of bins.
//
template<class T>
static void bin_counts(const Reference_Counted_Array::Array<T> &seq,
		       float min, float max, IArray &counts)
{
  int bins = counts.size();
  int *c = counts.values();

  float range = max - min;
  if (range == 0)
    return;

  T *data = seq.values();
  float scale = bins / range;

  int dim = seq.dimension();
  int m0 = 1, m1 = 1, m2 = 1, s0 = 0, s1 = 0, s2 = 0;
  if (dim == 1)
    { m2 = seq.size(0); s2 = seq.stride(0); }
  else if (dim == 2)
    { s1 = seq.stride(0); s2 = seq.stride(1);
      m1 = seq.size(0); m2 = seq.size(1); }
  else if (dim == 3)
    { s0 = seq.stride(0); s1 = seq.stride(1); s2 = seq.stride(2);
      m0 = seq.size(0); m1 = seq.size(1); m2 = seq.size(2); }

  int i = 0;
  for (int i0 = 0 ; i0 < m0 ; ++i0, i += s0 - m1*s1)
    for (int i1 = 0 ; i1 < m1 ; ++i1, i += s1 - m2*s2)
      for (int i2 = 0 ; i2 < m2 ; ++i2, i += s2)
	{
	  int b = static_cast<int> (scale * (data[i] - min));
	  if (b >= 0 && b < bins)
	    c[b] += 1;
	}
}

// ----------------------------------------------------------------------------
// Return a Numeric array of integers that counts the number of data values
// in a series of bins.
//
extern "C" PyObject *
bin_counts_py(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array seq;
  IArray counts;
  float min, max;
  const char *kwlist[] = {"array", "min", "max", "counts", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ffO&"),
				   (char **)kwlist,
				   parse_array, &seq,
				   &min, &max,
				   parse_writable_int_n_array, &counts))
    return NULL;

  if (seq.dimension() < 1 || seq.dimension() > 3)
    {
      PyErr_SetString(PyExc_TypeError,
		      "minimum_and_maximum(): array must be 1, 2, or 3 dimensional");
      return NULL;
    }

  if (!counts.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "bin_counts(): output array must be contiguous");
      return NULL;
    }

  call_template_function(bin_counts, seq.value_type(), (seq, min, max, counts));

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
template<class T>
static void high_count(const Reference_Counted_Array::Array<T> &d,
		       float level, int *n)
{
  T *data = d.values();
  int s0 = d.stride(0), s1 = d.stride(1), s2 = d.stride(2);
  int m0 = d.size(0), m1 = d.size(1), m2 = d.size(2);
  int i = 0, c = 0;
  for (int i0 = 0 ; i0 < m0 ; ++i0, i += s0 - m1*s1)
    for (int i1 = 0 ; i1 < m1 ; ++i1, i += s1 - m2*s2)
      for (int i2 = 0 ; i2 < m2 ; ++i2, i += s2)
	if (data[i] >= level)
	  c += 1;
  *n = c;
}

// ----------------------------------------------------------------------------
// Return count of elements where data is greater than or equal to a
// specified level.
//
extern "C" PyObject *
high_count_py(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array d;
  float level;
  const char *kwlist[] = {"array", "level", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&f"),
				   (char **)kwlist,
				   parse_3d_array, &d,
				   &level))
    return NULL;

  int n = 0;
  call_template_function(high_count, d.value_type(), (d, level, &n));
  return PyLong_FromLong(n);
}

// ----------------------------------------------------------------------------
//
template<class T>
static void high_indices(const Reference_Counted_Array::Array<T> &d,
			 float level, int *ijk)
{
  T *data = d.values();
  int s0 = d.stride(0), s1 = d.stride(1), s2 = d.stride(2);
  int m0 = d.size(0), m1 = d.size(1), m2 = d.size(2);
  int i = 0, c = 0;
  for (int i0 = 0 ; i0 < m0 ; ++i0, i += s0 - m1*s1)
    for (int i1 = 0 ; i1 < m1 ; ++i1, i += s1 - m2*s2)
      for (int i2 = 0 ; i2 < m2 ; ++i2, i += s2)
	if (data[i] >= level)
	  { ijk[c] = i2; ijk[c+1] = i1; ijk[c+2] = i0; c += 3; }
}

// ----------------------------------------------------------------------------
// Return indices (n by 3 array) where data is greater than or equal to a
// specified level.
//
extern "C" PyObject *
high_indices_py(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array d;
  float level;
  const char *kwlist[] = {"array", "level", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&f"),
				   (char **)kwlist,
				   parse_3d_array, &d,
				   &level))
    return NULL;

  int n = 0;
  call_template_function(high_count, d.value_type(), (d, level, &n));
  int *ijk;
  PyObject *indices = python_int_array(n, 3, &ijk);
  call_template_function(high_indices, d.value_type(), (d, level, ijk));
  return indices;
}

} // namespace Map_Cpp
