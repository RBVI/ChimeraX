// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject	

// #include <iostream>			// use std::cerr for debugging

#include "pythonarray.h"		// use array_from_python(), ...
#include "rcarray.h"			// use Array<T>, Numeric_Array
using Reference_Counted_Array::Numeric_Array;

namespace Volume_Display
{

typedef Reference_Counted_Array::Numeric_Array Color_Array;
typedef FArray Color_Float_Array; // 1-4 color components
typedef FArray RGBA_Float_Array;
typedef Reference_Counted_Array::Numeric_Array Index_Array; // 8 or 16-bit int
typedef FArray Transfer_Function;

// ----------------------------------------------------------------------------
//
extern "C" int parse_transfer_function(PyObject *arg, void *transfer_func);
static void transfer_function_colors(const Transfer_Function &transfer_func,
				     float bcf, float bcl,
				     RGBA_Float_Array &colormap,
				     int bins, int bin_step, bool blend);
static bool check_color_array_size(const Reference_Counted_Array::Untyped_Array &colors,
			    const Numeric_Array &data, int nc);
extern "C" int parse_colormap(PyObject *arg, void *cmap);
extern "C" int parse_float_colormap(PyObject *arg, void *cmap);
static bool colormap_value(float d, float bcf, float bcl,
			   float *cmap, int nc, int ncc, float *color);
static bool float_colormap(PyObject *py_colormap,
			   Color_Float_Array *carray,				
			   bool require_contiguous = false,
			   int nc = 0, int size_divisor = 0);
static bool colormap(PyObject *py_colormap, Color_Array *cmap,
		     bool require_contiguous = false,
		     int nc = 0, int size_divisor = 0);
static void resample_colormap(float bcf1, float bcl21,
			      const Color_Float_Array &cmap1,
			      float bcf2, float bcl2, Color_Float_Array &cmap2,
			      int bins, int bin_step, bool blend);

// ----------------------------------------------------------------------------
//
inline
static bool transfer_function_value(const Transfer_Function &transfer_func,
				    float value,
				    float *r, float *g, float *b, float *a)
{
  float *tf = transfer_func.values();
  int m6 = 6 * transfer_func.size(0);

  int j = 0;
  while (j < m6 && value >= tf[j])
    j += 6;

  if (j == 0)
    return false;

  if (j < m6)
    {
      float f0 = (tf[j] - value) / (tf[j] - tf[j-6]);
      float f1 = 1 - f0;
      float scale = f0*tf[j-5] + f1*tf[j+1];
      *r = scale * (f0*tf[j-4] + f1*tf[j+2]);
      *g = scale * (f0*tf[j-3] + f1*tf[j+3]);
      *b = scale * (f0*tf[j-2] + f1*tf[j+4]);
      *a = scale * (f0*tf[j-1] + f1*tf[j+5]);
    }
  else if (value == tf[j-6])
    {
      float scale = tf[j-5];
      *r = scale * tf[j-4];
      *g = scale * tf[j-3];
      *b = scale * tf[j-2];
      *a = scale * tf[j-1];
    }
  else
    return false;

  return true;
}

// ----------------------------------------------------------------------------
// RGBA array must be contiguous.
//
template <class T>
static void data_to_rgba(const Reference_Counted_Array::Array<T> &data,
			 const Transfer_Function &transfer_func,
			 RGBA_Float_Array &rgba, bool blend)
{
  Reference_Counted_Array::Array<T> cdata = data.contiguous_array();

  int n = cdata.size();
  T *d = cdata.values();
  float *rgba_values = rgba.values();
  float r = 0, g = 0, b = 0, a = 0;		// Avoid compiler warnings.
  for (int k = 0 ; k < n ; ++k)
    if (transfer_function_value(transfer_func, (float)d[k], &r, &g, &b, &a))
      {
	float *rgbak = rgba_values + 4*k;
	if (blend)
	  {
	    rgbak[0] += r;
	    rgbak[1] += g;
	    rgbak[2] += b;
	    rgbak[3] = 1 - (1-a) * (1-rgbak[3]);
	  }
	else
	  {
	    rgbak[0] = r;
	    rgbak[1] = g;
	    rgbak[2] = b;
	    rgbak[3] = a;
	  }
      }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *data_to_rgba(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array data, nrgba;
  Transfer_Function transfer_func;
  int iblend;
  const char *kwlist[] = {"data", "transfer_function", "rgba", "blend", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&OO&i"),
				   (char **)kwlist,
				   parse_3d_array, &data,
				   parse_transfer_function, &transfer_func,
				   parse_writable_4d_array, &nrgba, &iblend))
    return NULL;
  bool blend = iblend;

  RGBA_Float_Array rgba = nrgba;
  if (!check_color_array_size(rgba, data, 4))
    return NULL;

  call_template_function(data_to_rgba, data.value_type(),
			 (data, transfer_func, rgba, blend));

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
// RGBA and colormap arrays must be contiguous.
//
template <class T>
static void data_to_colormap_colors(const Reference_Counted_Array::Array<T> &data,
				    float bcf, float bcl,
				    const Color_Float_Array &colormap,
				    Color_Float_Array &colors, bool blend)
{
  if (bcl - bcf <= 0)
    return;

  Reference_Counted_Array::Array<T> cdata = data.contiguous_array();
  const Color_Float_Array &ccolormap = colormap.contiguous_array();

  int n = cdata.size();
  T *d = cdata.values();
  int nc = ccolormap.size(0), ncc = ccolormap.size(1), nca = ncc-1;
  float *cmap = ccolormap.values();
  float *color_values = colors.values();
  if (blend)
    {
      float *cmk = new float[ncc];
      for (int k = 0 ; k < n ; ++k)
	if (colormap_value((float)d[k], bcf, bcl, cmap, nc, ncc, cmk))
	  {
	    float *ck = color_values + ncc*k;
	    for (int c = 0 ; c < nca ; ++c)
	      ck[c] += cmk[c];	// color channel (r,g,b, or l)
	    if (ncc > 1)
	      ck[nca] = 1 - (1-cmk[nca]) * (1-ck[nca]); // Alpha
	  }
      delete [] cmk;
    }
  else
    for (int k = 0 ; k < n ; ++k)
      colormap_value((float)d[k], bcf, bcl, cmap, nc, ncc, color_values+ncc*k);
}

// ----------------------------------------------------------------------------
//
static bool colormap_value(float d, float bcf, float bcl,
			   float *cmap, int nc, int ncc, float *cmd)
{
  float c = (nc-1) * ((d - bcf) / (bcl - bcf));
  if (c < 0 || c > nc-1)
    return false;	// Out of colormap range.

  int ci = static_cast<int>(c);
  float *c0 = cmap + ncc*ci;
  float *c1 = (ci == nc-1 ? c0 : c0 + 4);
  float f = c - ci;
  for (int i = 0 ; i < ncc ; ++i)
    cmd[i] = (1-f)*c0[i] + f*c1[i];
  return true;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
data_to_colormap_colors(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array data;
  Color_Float_Array cmap, colors;
  float bcf, bcl;
  int iblend;
  const char *kwlist[] = {"data", "dmin", "dmax", "colormap", "colors",
			  "blend", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ffO&O&i"),
				   (char **)kwlist,
				   parse_3d_array, &data,
				   &bcf, &bcl,
				   parse_float_colormap, &cmap,
				   parse_writable_4d_array, &colors,
				   &iblend) ||
      !check_color_array_size(colors, data, cmap.size(1)))
    return NULL;
  bool blend = iblend;

  call_template_function(data_to_colormap_colors, data.value_type(),
			 (data, bcf, bcl, cmap, colors, blend));

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static bool check_color_array_size(const Reference_Counted_Array::Untyped_Array &colors,
				   const Numeric_Array &data, int nc)
{
  int d = data.dimension();
  if (d + 1 != colors.dimension())
    {
      PyErr_Format(PyExc_TypeError, "Color array dimension (%d) is not one more than data array dimension (%d).",
		   colors.dimension(), d);
      return false;
    }

  for (int k = 0 ; k < d ; ++k)
    if (data.size(k) != colors.size(k))
      {
	if (d == 1)
	  PyErr_Format(PyExc_TypeError, "Color array size %d does not match data array size %d",
		       colors.size(0), data.size(0));
	else if (d == 2)
	  PyErr_Format(PyExc_TypeError, "Color array size (%d,%d) does not match data array size (%d,%d)",
		       colors.size(0), colors.size(1), data.size(0), data.size(1));
	else
	  PyErr_Format(PyExc_TypeError, "Color array size (%d,%d,%d) does not match data array size (%d,%d,%d)",
		       colors.size(0), colors.size(1), colors.size(2), data.size(0), data.size(1), data.size(2));
	return false;
      }

  if (colors.size(d) != nc)
    {
      PyErr_Format(PyExc_TypeError, "Must have %d color components, got %d", nc, colors.size(d));
      return false;
    }

  if (!colors.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Color array is non-contiguous");
      return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
// Color and colormap arrays must be contiguous.
//
template <class T>
static void data_to_colors(const Reference_Counted_Array::Array<T> &data,
			   float dmin, float dmax, const Color_Array &colormap,
			   bool clamp, Color_Array &colors)
{
  if (dmax - dmin <= 0)
    return;

  Reference_Counted_Array::Array<T> cdata = data.contiguous_array();
  const Color_Array &ccolormap = colormap.contiguous_array();

  int n = cdata.size();
  T *d = cdata.values();
  int nc = ccolormap.size(0);
  if (nc == 0)
    clamp = false;
  float scale = nc / (dmax - dmin);
  int ncb = ccolormap.size(1) * ccolormap.element_size();
  bool data_is_index = (dmin == 0 && dmax == nc-1);
  if (ncb == 4)
    {
      // Optimize common case of 4 byte colors (8-bit rgba).
      unsigned int *cmap = (unsigned int *) ccolormap.values();
      unsigned int c0 = (clamp ? cmap[0] : 0);
      unsigned int c1 = (clamp ? cmap[nc-1] : 0);
      unsigned int *cv = (unsigned int *)colors.values();
      if (data_is_index)
	{
	  // Optimize common case where data value = color map index.
	  for (int k = 0 ; k < n ; ++k)
	    {
	      int i = (int)d[k];
	      cv[k] = (i >= 0 ? (i < nc ? cmap[i] : c1) : c0);
	    }
	}
      else
	{
	  for (int k = 0 ; k < n ; ++k)
	    {
	      int i = (int)((d[k]-dmin)*scale);
	      cv[k] =(i >= 0 ? (i < nc ? cmap[i] : c1) : c0);
	    }
	}
    }
  else if (ncb == 2)
    {
      // Optimize common case of 2 byte colors (8-bit luminance/alpha).
      unsigned short *cmap = (unsigned short *) ccolormap.values();
      unsigned short c0 = (clamp ? cmap[0] : 0);
      unsigned short c1 = (clamp ? cmap[nc-1] : 0);
      unsigned short *cv = (unsigned short *)colors.values();
      if (data_is_index)
	{
	  // Optimize common case where data value = color map index.
	  for (int k = 0 ; k < n ; ++k)
	    {
	      int i = (int)d[k];
	      cv[k] = (i >= 0 ? (i < nc ? cmap[i] : c1) : c0);
	    }
	}
      else
	{
	  for (int k = 0 ; k < n ; ++k)
	    {
	      int i = (int)((d[k]-dmin)*scale);
	      cv[k] = (i >= 0 ? (i < nc ? cmap[i] : c1) : c0);
	    }
	}
    }
  else if (ncb == 1)
    {
      // Optimize common case of 1 byte color (8-bit single opaque color)
      unsigned char *cmap = (unsigned char *) ccolormap.values();
      unsigned char c0 = (clamp ? cmap[0] : 0);
      unsigned char c1 = (clamp ? cmap[nc-1] : 0);
      unsigned char *cv = (unsigned char *)colors.values();
      if (data_is_index)
	{
	  // Optimize common case where data value = color map index.
	  for (int k = 0 ; k < n ; ++k)
	    {
	      int i = (int)d[k];
	      cv[k] = (i >= 0 ? (i < nc ? cmap[i] : c1) : c0);
	    }
	}
      else
	{
	  for (int k = 0 ; k < n ; ++k)
	    {
	      int i = (int)((d[k]-dmin)*scale);
	      cv[k] = (i >= 0 ? (i < nc ? cmap[i] : c1) : c0);
	    }
	}
    }
  else
    {
      char *cmap = (char *) ccolormap.values();
      char *c0 = (clamp ? cmap : (char *)0);
      char *c1 = (clamp ? cmap + (nc-1)*ncb : (char *)0);
      char *cv = (char *)colors.values();
      for (int k = 0 ; k < n ; ++k)
	{
	  int i = (int)(data_is_index ? d[k] : (d[k]-dmin)*scale);
	  int kc = ncb*k, ic = ncb*i;
	  if (i >= 0 && i < nc)
	    for (int c = 0 ; c < ncb ; ++c)
	      cv[kc+c] = cmap[ic+c];
	  else if (clamp)
	    if (i < 0)
	      for (int c = 0 ; c < ncb ; ++c)
		cv[kc+c] = c0[c];
	    else
	      for (int c = 0 ; c < ncb ; ++c)
		cv[kc+c] = c1[c];
	  else
	    for (int c = 0 ; c < ncb ; ++c)
	      cv[kc+c] = (char) 0;
	}
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
data_to_colors(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array data;
  Color_Array cmap;
  float dmin, dmax;
  int iclamp;
  PyObject *py_colors;
  const char *kwlist[] = {"data", "dmin", "dmax", "colormap", "clamp",
			  "colors", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ffO&iO"),
				   (char **)kwlist,
				   parse_array, &data,
				   &dmin, &dmax,
				   parse_colormap, &cmap,
				   &iclamp,
				   &py_colors))
    return NULL;

  bool clamp = iclamp;

  int d = data.dimension();

  bool allow_data_copy = false;
  Numeric_Array c;
  if (!array_from_python(py_colors, d+1, cmap.value_type(), &c, allow_data_copy))
    return NULL;

  Color_Array colors = c;
  if (!check_color_array_size(colors, data, cmap.size(1)))
    return NULL;

  call_template_function(data_to_colors, data.value_type(),
			 (data, dmin, dmax, cmap, clamp, colors));

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
colors_float_to_uint(PyObject *s, PyObject *args, PyObject *keywds)
{
  Color_Float_Array colors_float;
  Color_Array colors_uint;
  const char *kwlist[] = {"colors_float", "colors_uint", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_float_array, &colors_float,
				   parse_writable_array, &colors_uint))
    return NULL;

  if (colors_float.dimension() != colors_uint.dimension())
    {
      PyErr_SetString(PyExc_TypeError, "Float and uint array dimensions differ");
      return NULL;
    }
  if (colors_float.size() != colors_uint.size())
    {
      PyErr_SetString(PyExc_TypeError, "Float and uint array sizes differ");
      return NULL;
    }
  Color_Array::Value_Type t = colors_uint.value_type();
  if (t != colors_uint.Unsigned_Char && t != colors_uint.Unsigned_Short_Int)
    {
      PyErr_SetString(PyExc_TypeError, "Only uint8 or uint16 destination array supported");
      return NULL;
    }
  if (!colors_float.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Float color array is non-contiguous");
      return NULL;
    }
  if (!colors_uint.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "uint color array is non-contiguous");
      return NULL;
    }

  float *f = colors_float.values();
  int n = colors_float.size();
  if (t == colors_uint.Unsigned_Char)
    {
      unsigned char *i8 = (unsigned char *)colors_uint.values();
      for (int k = 0 ; k < n ; ++k)
	{
	  float c = f[k];
	  if (c > 1) c = 1;
	  else if (c < 0) c = 0;
	  i8[k] = (unsigned char)(255*c);
	}
    }
  else if (t == colors_uint.Unsigned_Short_Int)
    {
      unsigned short *i16 = (unsigned short *)colors_uint.values();
      for (int k = 0 ; k < n ; ++k)
	{
	  float c = f[k];
	  if (c > 1) c = 1;
	  else if (c < 0) c = 0;
	  i16[k] = (unsigned short)(65535*c);
	}
    }

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
inline void scale_8_bit_rgba(unsigned int *rgba, unsigned int s)
{
  const unsigned int b = 0xff;
  unsigned int c = *rgba;
  *rgba = ((((c&b) * (s&b)) >> 8) |
	   (((((c>>8)&b) * ((s>>8)&b)) >> 8) << 8) |
	   (((((c>>16)&b) * ((s>>16)&b)) >> 8) << 16) |
	   (((((c>>24)&b) * ((s>>24)&b)) >> 8) << 24));
}

// ----------------------------------------------------------------------------
//
template <class I>
static void index_colors(I *indices, int n, void *colormap, int nc,
			 void *colors, bool modulate)
{
  if (nc == 4)
    {
      // Optimize 4 byte colors (e.g. 8-bit BGRA).
      unsigned int *cm = (unsigned int *)colormap;
      unsigned int *c = (unsigned int *)colors;
      if (modulate)
	for (int i = 0 ; i < n ; ++i)
	  scale_8_bit_rgba(c+i, cm[indices[i]]);
      else
	for (int i = 0 ; i < n ; ++i)
	  c[i] = cm[indices[i]];
    }
  else if (nc == 2 && !modulate)
    {
      // Optimize 2 byte colors (e.g. 8-bit LA).
      unsigned short *cm = (unsigned short *)colormap;
      unsigned short *c = (unsigned short *)colors;
      for (int i = 0 ; i < n ; ++i)
	c[i] = cm[indices[i]];
    }
  else
    {
      unsigned char *cmap = (unsigned char *)colormap;
      unsigned char *cl = (unsigned char *)colors;
      if (modulate)
	for (int i = 0 ; i < n ; ++i)
	  {
	    unsigned char *ci = cl + i*nc, *cmi = cmap + indices[i]*nc;
	    for (int c = 0 ; c < nc ; ++c)
	      ci[c] = (((int)ci[c]*(int)cmi[c]) >> 8);
	  }
      else
	for (int i = 0 ; i < n ; ++i)
	  {
	    unsigned char *ci = cl + i*nc, *cmi = cmap + indices[i]*nc;
	    for (int c = 0 ; c < nc ; ++c)
	      ci[c] = cmi[c];
	  }
    }
}

// ----------------------------------------------------------------------------
// Colormap and colors arrays must be contiguous.
//
extern "C" PyObject *
indices_to_colors(PyObject *s, PyObject *args, PyObject *keywds)
{
  Index_Array indices;
  Color_Array cmap, colors;
  int imodulate = 0;
  const char *kwlist[] = {"indices", "colormap", "colors", "modulate", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&|i"),
				   (char **)kwlist,
				   parse_array, &indices,
				   parse_2d_array, &cmap,
				   parse_array, &colors,
				   &imodulate))
    return NULL;
  bool modulate = imodulate;

  if (cmap.value_type() != Color_Array::Unsigned_Char &&
      cmap.value_type() != Color_Array::Unsigned_Short_Int)
    {
      PyErr_SetString(PyExc_TypeError, "Colormap type must be uint8 or uint16.");
      return NULL;
    }
  if (cmap.size(1) < 1 || cmap.size(1) > 4)
    {
      PyErr_SetString(PyExc_TypeError, "Colormap second dimension size is not 1, 2, 3, or 4.");
      return NULL;
    }
  if (!cmap.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Colormap array is non-contiguous");
      return NULL;
    }
  if (colors.dimension() != indices.dimension()+1)
    {
      PyErr_Format(PyExc_TypeError, "Color array dimension %d must be one more than index array dimension %d",
		   colors.dimension(), indices.dimension());
      return NULL;
    }
  if (!check_color_array_size(colors, indices, cmap.size(1)))
    return NULL;
  if (colors.value_type() != Color_Array::Unsigned_Char &&
      colors.value_type() != Color_Array::Unsigned_Short_Int)
    {
      PyErr_SetString(PyExc_TypeError, "Color array type must be uint8 or uint16.");
      return NULL;
    }

  void *cmapv = cmap.values();
  void *ca = colors.values();
  Index_Array ci = indices.contiguous_array();
  int n = ci.size(), nc = cmap.size(1)*cmap.element_size();
  Index_Array::Value_Type it = indices.value_type();
  if (it == ci.Unsigned_Char || it == ci.Signed_Char || it == ci.Char)
    index_colors((unsigned char *)ci.values(), n, cmapv, nc, ca, modulate);
  else if (it == ci.Unsigned_Short_Int || it == ci.Short_Int)
    index_colors((unsigned short *)ci.values(), n, cmapv, nc, ca, modulate);
  else if (it == ci.Unsigned_Int || it == ci.Int)
    index_colors((unsigned int *)ci.values(), n, cmapv, nc, ca, modulate);
  else
    {
      PyErr_SetString(PyExc_TypeError, "Index array type is not 8, 16, or 32-bit integers");
      return NULL;
    }

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
// Returned transfer function array is guaranteed to be contiguous.
//
extern "C" int parse_transfer_function(PyObject *arg, void *transfer_func)
{
  Numeric_Array tf;
  if (!array_from_python(arg, 2, Numeric_Array::Float, &tf))
    return 0;
  Transfer_Function tfunc = tf;
  tfunc = tf.contiguous_array();

  if (tfunc.size(0) > 0 && tfunc.size(1) != 6)
    {
      PyErr_Format(PyExc_TypeError, "Transfer function array second dimension must have size 6 "
		   "(data_value,intensity_scale,r,g,b,a), got size %d", tfunc.size(1));
      return 0;
    }
  *static_cast<Transfer_Function *>(transfer_func) = tfunc;

  return 1;
}

// ----------------------------------------------------------------------------
//
namespace
{  // Tru64 compiler failed to match this template when declared static.
   // Use unnamed namespace instead.
  
template <class T, class I>
void data_to_index(const Reference_Counted_Array::Array<T> &data,
		   float bcf, float bcl, int bins, int bin_step,
		   I *indices, bool add)
{
  if (bins == 0)
    return;

  float range = bcl - bcf;
  if (range == 0)
    return;

  float br = bins / range;
  I bmax = (I) (bins-1);

  Reference_Counted_Array::Array<T> cdata = data.contiguous_array();

  int n = cdata.size();
  T *d = cdata.values();
  I b;
  if (bin_step == 1 && !add)	// Optimize most common case.
    for (int k = 0 ; k < n ; ++k)
      {
	float v = d[k] - bcf;
	if (v < 0) b = 0;
	else
	  {
	    b = (I) (br * v);
	    if (b > bmax) b = bmax;
	  }
	indices[k] = b;
      }
  else
    for (int k = 0 ; k < n ; ++k)
      {
	float v = d[k] - bcf;
	if (v < 0) b = 0;
	else
	  {
	    b = (I) (br * v);
	    if (b > bmax) b = bmax;
	  }
	if (add)
	  indices[k] += (I) (b * bin_step);
	else
	  indices[k] = (I) (b * bin_step);
      }
}

}

// ----------------------------------------------------------------------------
//
template <class T>
static void data_to_bin_index(const Reference_Counted_Array::Array<T> &data,
			      float bcf, float bcl, int bins, int bin_step,
			      Index_Array &index_values, bool add)
{
  if (index_values.value_type() == index_values.Unsigned_Short_Int)
    {
      Reference_Counted_Array::Array<unsigned short> i = index_values;
      data_to_index(data, bcf, bcl, bins, bin_step, i.values(), add);
    }
  else if (index_values.value_type() == index_values.Unsigned_Char)
    {
      Reference_Counted_Array::Array<unsigned char> i = index_values;
      data_to_index(data, bcf, bcl, bins, bin_step, i.values(), add);
    }
}

// ----------------------------------------------------------------------------
// Index values array must be a contiguous Numeric Python array.
//
extern "C" PyObject *
data_to_bin_index(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array data;
  Index_Array index_values;
  float bcf, bcl;
  int bins, bin_step, iadd;
  const char *kwlist[] = {"data", "bcfirst", "bclast", "bins", "bin_step",
			  "index_values", "add", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ffiiO&i"),
				   (char **)kwlist,
				   parse_3d_array, &data,
				   &bcf, &bcl,
				   &bins, &bin_step,
				   parse_writable_3d_array, &index_values,
				   &iadd))
    return NULL;
  bool add = iadd;

  if (index_values.value_type() != Numeric_Array::Unsigned_Short_Int &&
      index_values.value_type() != Numeric_Array::Unsigned_Char)
    {
      PyErr_SetString(PyExc_TypeError, "Index values array type must be uint8 or uint16");
      return NULL;
    }

  if (data.size(0) != index_values.size(0) ||
      data.size(1) != index_values.size(1) ||
      data.size(2) != index_values.size(2))
    {
      PyErr_Format(PyExc_TypeError, "Index array size (%d,%d,%d) does not match data array size (%d,%d,%d)",
		   index_values.size(0), index_values.size(1), index_values.size(2),
		   data.size(0), data.size(1), data.size(2));
      return NULL;
    }

  if (!index_values.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Index values array is non-contiguous");
      return NULL;
    }

  call_template_function(data_to_bin_index, data.value_type(),
			 (data, bcf, bcl, bins, bin_step, index_values, add));

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
transfer_function_colormap(PyObject *s, PyObject *args, PyObject *keywds)
{
  Transfer_Function transfer_func;
  PyObject *py_colormap;
  float bcf, bcl; 
  int bins = 0, bin_step = 1, iblend = 0;
  const char *kwlist[] = {"transfer_function", "bcfirst", "bclast",
			  "colormap", "bins", "bin_step", "blend", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ffO|iii"),
				   (char **)kwlist,
				   parse_transfer_function, &transfer_func,
				   &bcf, &bcl,
				   &py_colormap,
				   &bins, &bin_step, &iblend))
    return NULL;
  bool blend = iblend;

  RGBA_Float_Array cmap;
  if (!float_colormap(py_colormap, &cmap, true, 4, bins * bin_step))
    return NULL;

  if (bins == 0)
    bins = cmap.size(0);

  transfer_function_colors(transfer_func, bcf, bcl, cmap, bins, bin_step, blend);

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_float_colormap(PyObject *arg, void *cmap)
{
  Color_Array cm;
  if (!colormap(arg, &cm, false, 0, 0))
    return 0;
  if (cm.value_type() != Color_Array::Float)
    {
      PyErr_SetString(PyExc_TypeError, "Colormap must have float values");
      return 0;
    }
  *static_cast<Color_Float_Array *>(cmap) = cm;
  return 1;
}

// ----------------------------------------------------------------------------
//
extern "C" int parse_colormap(PyObject *arg, void *cmap)
{
  return colormap(arg, static_cast<Color_Array *>(cmap), false, 0, 0) ? 1 : 0;
}

// ----------------------------------------------------------------------------
//
static bool float_colormap(PyObject *py_colormap,
			   Color_Float_Array *carray,				
			   bool require_contiguous,
			   int nc, int size_divisor)
{
  Color_Array cmap;
  if (!colormap(py_colormap, &cmap, require_contiguous, nc, size_divisor))
    return false;
  if (cmap.value_type() != Color_Array::Float)
    {
      PyErr_Format(PyExc_TypeError, "Colormap must have float values, got %s",
		   cmap.value_type_name(cmap.value_type()));
      return false;
    }
  *carray = cmap;
  return true;
}

// ----------------------------------------------------------------------------
//
static bool colormap(PyObject *py_colormap, Color_Array *carray,
		     bool require_contiguous, int nc, int size_divisor)
{
  bool allow_data_copy = false;
  Numeric_Array cm;
  if (!array_from_python(py_colormap, 2, &cm, allow_data_copy))
    return false;
  Color_Array cmap = cm;
  if (nc > 0 && cmap.size(1) != nc)
    {
      PyErr_Format(PyExc_TypeError, "The 2nd dimension of colormap array must have size %d, got %d",
		   nc, cmap.size(1));
      return false;
    }
  if (size_divisor > 0 && cmap.size(0) % size_divisor != 0)
    {
      PyErr_Format(PyExc_TypeError, "Colormap size (%d) must be a multiple of %d",
		   cmap.size(0), size_divisor);
      return false;
    }
  if (require_contiguous && !cmap.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Colormap array is non-contiguous");
      return false;
    }
  *carray = cmap;

  return true;
}

// ----------------------------------------------------------------------------
//
static void transfer_function_colors(const Transfer_Function &transfer_func,
				     float bcf, float bcl,
				     RGBA_Float_Array &colormap,
				     int bins, int bin_step, bool blend)
{
  int bb_step = colormap.size(0) / (bins * bin_step);
  float *cmap = colormap.values();
  for (int bi = 0 ; bi < bins ; ++bi)
    {
      float bc = (bi == 0 ? bcf :
		  (bi == bins-1 ? bcl :
		   bcf + (bcl - bcf) * bi/(bins-1.0)));
      float r = 0, g = 0, b = 0, a = 0;		// Avoid compiler warnings.
      if (transfer_function_value(transfer_func, bc, &r, &g, &b, &a))
	{
	  for (int i = 0 ; i < bin_step ; ++i)
	    for (int j = 0 ; j < bb_step ; ++j)
	      {
		int offset = 4 * (i + bi*bin_step + j*bins*bin_step);
		float *cmap_rgba = cmap + offset;
		if (blend)
		  {
		    cmap_rgba[0] += r;
		    cmap_rgba[1] += g;
		    cmap_rgba[2] += b;
		    cmap_rgba[3] = 1 - (1-a) * (1-cmap_rgba[3]);
		  }
		else
		  {
		    cmap_rgba[0] = r;
		    cmap_rgba[1] = g;
		    cmap_rgba[2] = b;
		    cmap_rgba[3] = a;
		  }
	      }
	}
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
resample_colormap(PyObject *s, PyObject *args, PyObject *keywds)
{
  Color_Float_Array cmap1;
  float bcf1, bcl1, bcf2, bcl2;
  PyObject *py_colormap2;
  int bins = 0, bin_step = 1, iblend = 0;
  const char *kwlist[] = {"bcfirst1", "bclast1", "colormap1",
			  "bcfirst2", "bclast2", "colormap2",
			  "bins", "bin_step", "blend", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("ffO&ffO|iii"),
				   (char **)kwlist,
				   &bcf1, &bcl1,
				   parse_float_colormap, &cmap1,
				   &bcf2, &bcl2, &py_colormap2,
				   &bins, &bin_step, &iblend))
    return NULL;
  bool blend = iblend;

  Color_Float_Array cmap2;
  if (!float_colormap(py_colormap2, &cmap2, true, 0, bins * bin_step))
    return NULL;
  if (cmap2.size(1) != cmap1.size(1))
    {
      PyErr_SetString(PyExc_TypeError, "Number of colors components doesn't match.");
      return NULL;
    }

  if (bins == 0)
    bins = cmap2.size(0);

  resample_colormap(bcf1, bcl1, cmap1, bcf2, bcl2, cmap2,
		    bins, bin_step, blend);

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static void resample_colormap(float bcf1, float bcl1,
			      const Color_Float_Array &cmap1,
			      float bcf2, float bcl2, Color_Float_Array &cmap2,
			      int bins, int bin_step, bool blend)
{
  Color_Float_Array ccmap1 = cmap1.contiguous_array();
  int nc1 = ccmap1.size(0), ncc = ccmap1.size(1), nca = ncc-1;
  float *cv1 = ccmap1.values();
    
  int bb_step = cmap2.size(0) / (bins * bin_step);
  float *cmap = cmap2.values();
  float *cmc = new float[ncc];
  for (int bi = 0 ; bi < bins ; ++bi)
    {
      float bc = (bi == 0 ? bcf2 :
		  (bi == bins-1 ? bcl2 :
		   bcf2 + (bcl2 - bcf2) * bi/(bins-1.0)));
      if (colormap_value(bc, bcf1, bcl1, cv1, nc1, ncc, cmc))
	{
	  for (int i = 0 ; i < bin_step ; ++i)
	    for (int j = 0 ; j < bb_step ; ++j)
	      {
		int offset = ncc * (i + bi*bin_step + j*bins*bin_step);
		float *cmap_rgba = cmap + offset;
		if (blend)
		  {
		    for (int k = 0 ; k < nca ; ++k)
		      cmap_rgba[k] += cmc[k];
		    if (ncc > 1) // Alpha
		      cmap_rgba[nca] = 1 - (1-cmc[nca]) * (1-cmap_rgba[nca]);
		  }
		else
		  for (int k = 0 ; k < ncc ; ++k)
		    cmap_rgba[k] = cmc[k];
	      }
	}
    }
  delete [] cmc;
}

}	// end of namespace Image_3d
