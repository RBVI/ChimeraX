// ----------------------------------------------------------------------------
// General array operations.
//
#include <Python.h>			// use PyObject
#include <vector>			// use std::vector

#include "pythonarray.h"		// use parse_string_array()
#include "rcarray.h"			// use CArray

namespace Image_3d
{

// ----------------------------------------------------------------------------
//
static void value_ranges(const char *sa, int n, int len, long stride, std::vector<int> &ranges)
{
  int s = 0;
  const char *ss = sa;
  for (int i = 0 ; i < n ; ++i)
    {
      const char *si = sa + i*stride;
      if (strncmp(si, ss, len) != 0)
	{
	  ranges.push_back(s);
	  ranges.push_back(i);
	  s = i;
	  ss = si;
	}
    }
  ranges.push_back(s);
  ranges.push_back(n);
}

// ----------------------------------------------------------------------------
// Return integer index ranges for runs of the same string array value.
//
extern "C" PyObject *
value_ranges(PyObject *s, PyObject *args, PyObject *keywds)
{
  CArray c;
  const char *kwlist[] = {"array", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_string_array, &c))
    return NULL;

  if (c.dimension() != 2)
    {
      PyErr_SetString(PyExc_TypeError,
		      "value_ranges(): array must be 2 dimensional");
      return NULL;
    }

  char *ca = c.values();
  int n = c.size(0), len = c.size(1);
  long st = c.stride(0);
  std::vector<int> ranges;
  value_ranges(ca, n, len, st, ranges);
  PyObject *r = c_array_to_python(ranges.data(), ranges.size()/2, 2);
  return r;
}

// -----------------------------------------------------------------------------
// Find intervals of contiguous integer values (i,i+1,i+2,...,i+k) in an array of increasing integer values.
// Do not include intervals of length 1.  Return pairs of a initial and final index from the input array.
//
static void contiguous_intervals(const int *a, int n, int stride, std::vector<int> &intervals)
{
  int s = 0, e;
  for (e = 0 ; e+1 < n ; ++e, a += stride)
    if (a[stride] != a[0]+1)
      {
	if (e > s)
	  {
	    intervals.push_back(s);
	    intervals.push_back(e);
	  }
	s = e+1;
      }
  if (e > s)
    {
      intervals.push_back(s);
      intervals.push_back(e);
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
contiguous_intervals(PyObject *s, PyObject *args, PyObject *keywds)
{
  IArray a;
  const char *kwlist[] = {"array", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_int_n_array, &a))
    return NULL;

  int *aa = a.values();
  int n = a.size(), st = a.stride(0);
  std::vector<int> intervals;
  contiguous_intervals(aa, n, st, intervals);
  PyObject *r = c_array_to_python(intervals.data(), intervals.size()/2, 2);
  return r;
}

// -----------------------------------------------------------------------------
// Find intervals where bool array mask is true in index interval i1 to i2.
//
static void mask_intervals(const char *mask, int n, int stride, int i1, int i2,
			   std::vector<int> &intervals)
{
  int s = 0;
  bool in = false;
  for (long i = i1 ; i <= i2 ; ++i)
    if (in)
      {
	if (! mask[i*stride])
	  {
	    intervals.push_back(s);
	    intervals.push_back(i-1);
	    in = false;
	  }
      }
    else if (mask[i*stride])
      {
        s = i;
	in = true;
      }

  if (in)
    {
      intervals.push_back(s);
      intervals.push_back(i2);
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
mask_intervals(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array a;
  int i1, i2;
  const char *kwlist[] = {"array", "imin", "imax", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ii"),
				   (char **)kwlist,
				   parse_1d_array, &a,
				   &i1, &i2))
    return NULL;

  Numeric_Array::Value_Type t = a.value_type();
  if (t != Numeric_Array::Char && t != Numeric_Array::Signed_Char && t != Numeric_Array::Unsigned_Char)
    {
      PyErr_SetString(PyExc_TypeError, "mask_intervals(): array must have char type");
      return NULL;
    }

  const char *aa = static_cast<const char *>(a.values());
  int n = a.size(), st = a.stride(0);
  std::vector<int> intervals;
  mask_intervals(aa, n, st, i1, i2, intervals);
  PyObject *r = c_array_to_python(intervals.data(), intervals.size()/2, 2);
  return r;
}

} // namespace Image_3d
