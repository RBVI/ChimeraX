// ----------------------------------------------------------------------------
// Blend images for motion blur.
//
#include <Python.h>			// use PyObject
#include <math.h>			// use ceil, floor

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use Numeric_Array, Array<T>

// ----------------------------------------------------------------------------
//
template<class T>
static void blend_colors(float f, const Reference_Counted_Array::Array<T> &m1,
			 const Reference_Counted_Array::Array<T> &m2,
			 const Reference_Counted_Array::Array<T> &bgcolor,
			 float alpha,
			 const Reference_Counted_Array::Array<T> &m)
			   
{
  long n = m.size();
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
	}
      v[k+3] = a;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *blur_blend_images(PyObject *s, PyObject *args, PyObject *keywds)
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
  bgcolor = array_from_python(bgcolor_py, 1, m.value_type());
  if (bgcolor.size() != 3 || !bgcolor.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "blend_images: bgcolor must be contiguous 3 element array");
      return NULL;
    }
  call_template_function(blend_colors, m.value_type(), (f, m1, m2, bgcolor, alpha, m));

  Py_INCREF(Py_None);
  return Py_None;
}
