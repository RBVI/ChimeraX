// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Python wrapper for _filter module.
//
#include <Python.h>				// use PyObject

#include <arrays/rcarray.h>			// Use Numeric_Array
#include <arrays/pythonarray.h>			// use parse_*()

using namespace Reference_Counted_Array;

namespace Map_Cpp
{

// ----------------------------------------------------------------------------
//
template <class T>
void moments(const T *data, const int *size,
	     double mo2[3][3], double mo1[3], double *mo0)
{
  int s0 = size[0], s1 = size[1], s2 = size[2];
  int st0 = s1*s2, st1 = s2;

  double m = 0, m0 = 0, m1 = 0, m2 = 0, m00 = 0, m01 = 0, m02 = 0, m11 = 0, m12 = 0, m22 = 0;

  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1)
      for (int i2 = 0 ; i2 < s2 ; ++i2)
	{
	  int i = i0*st0 + i1*st1 + i2;
	  T v = data[i];
	  m00 += i0*i0*v;
	  m11 += i1*i1*v;
	  m22 += i2*i2*v;
	  m01 += i0*i1*v;
	  m02 += i0*i2*v;
	  m12 += i1*i2*v;
	  m0 += i0*v;
	  m1 += i1*v;
	  m2 += i2*v;
	  m += v;
	}
  mo2[0][0] = m00; mo2[1][1] = m11; mo2[2][2] = m22;
  mo2[0][1] = mo2[1][0] = m01;
  mo2[0][2] = mo2[2][0] = m02;
  mo2[1][2] = mo2[2][1] = m12;
  mo1[0] = m0; mo1[1] = m1; mo1[2] = m2;
  *mo0 = m;
}

// ----------------------------------------------------------------------------
//
template <class T>
void affine_scale(T *data, const int *size, double c, double *u, bool invert)
{
  int s0 = size[0], s1 = size[1], s2 = size[2];
  int st0 = s1*s2, st1 = s2;
  double u0 = u[0], u1 = u[1], u2 = u[2];

  if (invert)
    {
      for (int i0 = 0 ; i0 < s0 ; ++i0)
	for (int i1 = 0 ; i1 < s1 ; ++i1)
	  for (int i2 = 0 ; i2 < s2 ; ++i2)
	    {
	      int i = i0*st0 + i1*st1 + i2;
	      data[i] /= c + u0*i0 + u1*i1 + u2*i2;
	    }
    }
  else
    {
      for (int i0 = 0 ; i0 < s0 ; ++i0)
	for (int i1 = 0 ; i1 < s1 ; ++i1)
	  for (int i2 = 0 ; i2 < s2 ; ++i2)
	    {
	      int i = i0*st0 + i1*st1 + i2;
	      data[i] *= c + u0*i0 + u1*i1 + u2*i2;
	    }
    }
}

// ----------------------------------------------------------------------------
//
template <class T>
void moments(const Array<T> &data, double m2[3][3], double m1[3], double *m0)
{
  const Array<T> dc = data.contiguous_array();
  moments(dc.values(), data.sizes(), m2, m1, m0);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *moments_py(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array data;
  const char *kwlist[] = {"data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_3d_array, &data))
    return NULL;

  double m2[3][3], m1[3], m0;
  call_template_function(moments, data.value_type(),
			 (data, m2, m1, &m0));

  PyObject *m2_py = c_array_to_python(&(m2[0][0]), 3, 3);
  PyObject *m1_py = c_array_to_python(&(m1[0]), 3);
  PyObject *m0_py = PyFloat_FromDouble(m0);
  PyObject *t = python_tuple(m2_py, m1_py, m0_py);

  return t;
}

// ----------------------------------------------------------------------------
//
template <class T>
void affine_scale(const Array<T> &data, double c, double u[3], bool invert)
{
  Array<T> dc = data.contiguous_array();
  affine_scale(dc.values(), data.sizes(), c, &u[0], invert);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *affine_scale_py(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array data;
  double c, u[3];
  bool invert = false;
  const char *kwlist[] = {"data", "c", "u", "invert", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&dO&|O&"),
				   (char **)kwlist,
				   parse_3d_array, &data, &c,
				   parse_double_3_array, &u[0],
				   parse_bool, &invert))
    return NULL;

  call_template_function(affine_scale, data.value_type(),
  			 (data, c, u, invert));

  Py_INCREF(Py_None);
  return Py_None;
}

} // end of namespace Map_Cpp
