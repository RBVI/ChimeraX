// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject	

// #include <iostream>			// use std::cerr for debugging
#include <stdexcept>			// use std::runtime_error
#include <sstream>			// use std::ostringstream

#include "pythonarray.h"		// use array_from_python(), ...
#include "rcarray.h"			// use Array<T>, Numeric_Array
using Reference_Counted_Array::Numeric_Array;

namespace Volume_Display
{

typedef Reference_Counted_Array::Numeric_Array Color_Array;
typedef Reference_Counted_Array::Array<float> Color_Float_Array; // 1-4 color components
typedef Reference_Counted_Array::Array<float> RGBA_Float_Array;
typedef Reference_Counted_Array::Numeric_Array Index_Array; // 8 or 16-bit int
typedef Reference_Counted_Array::Array<float> Transfer_Function;

// ----------------------------------------------------------------------------
//
static Transfer_Function transfer_function(PyObject *py_transfer_func);
static void transfer_function_colors(const Transfer_Function &transfer_func,
				     float bcf, float bcl,
				     RGBA_Float_Array &colormap,
				     int bins, int bin_step, bool blend);
static void check_color_array_size(const Reference_Counted_Array::Untyped_Array &colors,
			    const Numeric_Array &data, int nc);
static bool colormap_value(float d, float bcf, float bcl,
			   float *cmap, int nc, int ncc, float *color);
static Color_Float_Array float_colormap(PyObject *py_colormap,
					bool require_contiguous = false,
					int nc = 0, int size_divisor = 0);
static Color_Array colormap(PyObject *py_colormap,
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
  PyObject *py_data, *py_transfer_func, *py_rgba;
  int iblend;
  const char *kwlist[] = {"data", "transfer_function", "rgba", "blend", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OOOi"),
				   (char **)kwlist, &py_data,
				   &py_transfer_func, &py_rgba, &iblend))
    return NULL;
  bool blend = iblend;

  try
    {
      Numeric_Array data = array_from_python(py_data, 3);

      Transfer_Function transfer_func = transfer_function(py_transfer_func);

      bool allow_data_copy = false;
      RGBA_Float_Array rgba = array_from_python(py_rgba, 4,
						Numeric_Array::Float,
						allow_data_copy);
      check_color_array_size(rgba, data, 4);

      call_template_function(data_to_rgba, data.value_type(),
			     (data, transfer_func, rgba, blend));
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

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
  PyObject *py_data, *py_colormap, *py_colors;
  float bcf, bcl;
  int iblend;
  const char *kwlist[] = {"data", "dmin", "dmax", "colormap", "colors",
			  "blend", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OffOOi"),
				   (char **)kwlist, &py_data, &bcf, &bcl,
				   &py_colormap, &py_colors, &iblend))
    return NULL;
  bool blend = iblend;

  try
    {
      Numeric_Array data = array_from_python(py_data, 3);

      Color_Float_Array cmap = float_colormap(py_colormap);

      bool allow_data_copy = false;
      Color_Float_Array colors = array_from_python(py_colors, 4,
						   Numeric_Array::Float,
						   allow_data_copy);
      check_color_array_size(colors, data, cmap.size(1));

      call_template_function(data_to_colormap_colors, data.value_type(),
			     (data, bcf, bcl, cmap, colors, blend));
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static void check_color_array_size(const Reference_Counted_Array::Untyped_Array &colors,
				   const Numeric_Array &data, int nc)
{
  int d = data.dimension();
  if (d + 1 != colors.dimension())
    {
      std::ostringstream msg;
      msg << "Color array dimension (" << colors.dimension()
	  << ") is not one more than data array dimension (" << d << ").";
      throw std::runtime_error(msg.str());
    }

  for (int k = 0 ; k < d ; ++k)
    if (data.size(k) != colors.size(k))
      {
	std::ostringstream msg;
	msg << "Color array size";
	for (int i = 0 ; i < d ; ++i) msg << " " << colors.size(i);
	msg << " does not match data array size";
	for (int i = 0 ; i < d ; ++i) msg << " " << data.size(i);
	throw std::runtime_error(msg.str());
      }

  if (colors.size(d) != nc)
    {
      std::ostringstream msg;
      msg << "Must have " << nc
	  << " color components, got " << colors.size(d);
      throw std::runtime_error(msg.str());
    }

  if (!colors.is_contiguous())
    throw std::runtime_error("Color array is non-contiguous");
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
  PyObject *py_data, *py_colormap, *py_colors;
  float dmin, dmax;
  int iclamp;
  const char *kwlist[] = {"data", "dmin", "dmax", "colormap", "clamp",
			  "colors", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OffOiO"),
				   (char **)kwlist, &py_data, &dmin, &dmax,
				   &py_colormap, &iclamp, &py_colors))
    return NULL;
  bool clamp = iclamp;

  try
    {
      Numeric_Array data = array_from_python(py_data, 0);
      int d = data.dimension();

      Color_Array cmap = colormap(py_colormap);

      bool allow_data_copy = false;
      Color_Array colors = array_from_python(py_colors, d+1, cmap.value_type(),
					     allow_data_copy);
      check_color_array_size(colors, data, cmap.size(1));

      call_template_function(data_to_colors, data.value_type(),
			     (data, dmin, dmax, cmap, clamp, colors));
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
colors_float_to_uint(PyObject *s, PyObject *args, PyObject *keywds)
{
  PyObject *py_colors_float, *py_colors_uint;
  const char *kwlist[] = {"colors_float", "colors_uint", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OO"),
				   (char **)kwlist, &py_colors_float,
				   &py_colors_uint))
    return NULL;

  try 
    {
      Color_Float_Array fc = array_from_python(py_colors_float, 0,
					       Numeric_Array::Float);

      bool allow_data_copy = false;
      Color_Array ic = array_from_python(py_colors_uint, 0, allow_data_copy);
      if (fc.dimension() != ic.dimension())
	throw std::runtime_error("Float and uint array dimensions differ");
      if (fc.size() != ic.size())
	throw std::runtime_error("Float and uint array sizes differ");
      Color_Array::Value_Type t = ic.value_type();
      if (t != ic.Unsigned_Char && t != ic.Unsigned_Short_Int)
	throw std::runtime_error("Only uint8 or uint16 destination array supported");
      if (!fc.is_contiguous())
	throw std::runtime_error("Float color array is non-contiguous");
      if (!ic.is_contiguous())
	throw std::runtime_error("UInt8 color array is non-contiguous");

      float *f = fc.values();
      int n = fc.size();
      if (t == ic.Unsigned_Char)
	{
	  unsigned char *i8 = (unsigned char *)ic.values();
	  for (int k = 0 ; k < n ; ++k)
	    {
	      float c = f[k];
	      if (c > 1) c = 1;
	      else if (c < 0) c = 0;
	      i8[k] = (unsigned char)(255*c);
	    }
	}
      else if (t == ic.Unsigned_Short_Int)
	{
	  unsigned short *i16 = (unsigned short *)ic.values();
	  for (int k = 0 ; k < n ; ++k)
	    {
	      float c = f[k];
	      if (c > 1) c = 1;
	      else if (c < 0) c = 0;
	      i16[k] = (unsigned short)(65535*c);
	    }
	}
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
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
  PyObject *py_indices, *py_colormap, *py_colors;
  int imodulate = 0;
  const char *kwlist[] = {"indices", "colormap", "colors", "modulate", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OOO|i"),
				   (char **)kwlist, &py_indices,
				   &py_colormap, &py_colors, &imodulate))
    return NULL;
  bool modulate = imodulate;

  try
    {
      Index_Array indices = array_from_python(py_indices, 0);

      Color_Array cmap = array_from_python(py_colormap, 2);
      if (cmap.value_type() != Color_Array::Unsigned_Char &&
	  cmap.value_type() != Color_Array::Unsigned_Short_Int)
	throw std::runtime_error("Colormap type must be uint8 or uint16.");
      if (cmap.size(1) < 1 || cmap.size(1) > 4)
	throw std::runtime_error("Colormap second dimension size is not 1, 2, 3, or 4.");
      if (!cmap.is_contiguous())
	throw std::runtime_error("Colormap array is non-contiguous");

      bool allow_data_copy = false;
      Color_Array colors = array_from_python(py_colors, indices.dimension()+1,
					     allow_data_copy);
      check_color_array_size(colors, indices, cmap.size(1));
      if (colors.value_type() != Color_Array::Unsigned_Char &&
	  colors.value_type() != Color_Array::Unsigned_Short_Int)
	throw std::runtime_error("Color array type must be uint8 or uint16.");

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
	throw std::runtime_error("Index array type is not 8, 16, or 32-bit integers");
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
// Returned transfer function array is guaranteed to be contiguous.
//
static Transfer_Function transfer_function(PyObject *py_transfer_func)
{
  Transfer_Function transfer_func = array_from_python(py_transfer_func, 2,
						      Numeric_Array::Float);
  transfer_func = transfer_func.contiguous_array();

  if (transfer_func.size(0) > 0 && transfer_func.size(1) != 6)
    {
      std::ostringstream msg;
      msg << "Transfer function array second dimension must have size 6 "
	  << "(data_value,intensity_scale,r,g,b,a), got size "
	  << transfer_func.size(1);
      throw std::runtime_error(msg.str());
    }

  return transfer_func;
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
  PyObject *py_data, *py_index_values;
  float bcf, bcl;
  int bins, bin_step, iadd;
  const char *kwlist[] = {"data", "bcfirst", "bclast", "bins", "bin_step",
			  "index_values", "add", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OffiiOi"),
				   (char **)kwlist, &py_data, &bcf, &bcl,
				   &bins, &bin_step, &py_index_values, &iadd))
    return NULL;
  bool add = iadd;

  try
    {
      Numeric_Array data = array_from_python(py_data, 3);
    
      bool allow_data_copy = false;
      Index_Array index_values = array_from_python(py_index_values, 3,
						   allow_data_copy);

      if (index_values.value_type() != Numeric_Array::Unsigned_Short_Int &&
	  index_values.value_type() != Numeric_Array::Unsigned_Char)
	throw std::runtime_error("Index values array type must be uint8 or uint16");

      if (data.size(0) != index_values.size(0) ||
	  data.size(1) != index_values.size(1) ||
	  data.size(2) != index_values.size(2))
	{
	  std::ostringstream msg;
	  msg << "Index array size ("
	      << index_values.size(0) << ","
	      << index_values.size(1)  << ","
	      << index_values.size(2)
	      << ")	does not match data array size ("
	      << data.size(0) << "," << data.size(1)  << "," << data.size(2)
	      << ")";
	  throw std::runtime_error(msg.str());
	}

      if (!index_values.is_contiguous())
	throw std::runtime_error("Index values array is non-contiguous");

      call_template_function(data_to_bin_index, data.value_type(),
			     (data, bcf, bcl, bins, bin_step, index_values, add));
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
transfer_function_colormap(PyObject *s, PyObject *args, PyObject *keywds)
{
  PyObject *py_transfer_func, *py_colormap;
  float bcf, bcl; 
  int bins = 0, bin_step = 1, iblend = 0;
  const char *kwlist[] = {"transfer_function", "bcfirst", "bclast",
			  "colormap", "bins", "bin_step", "blend", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OffO|iii"),
				   (char **)kwlist, &py_transfer_func,
				   &bcf, &bcl, &py_colormap,
				   &bins, &bin_step, &iblend))
    return NULL;
  bool blend = iblend;

  try
    {
      Transfer_Function transfer_func = transfer_function(py_transfer_func);

      RGBA_Float_Array cmap = float_colormap(py_colormap, true, 4, bins * bin_step);

      if (bins == 0)
	bins = cmap.size(0);

      transfer_function_colors(transfer_func, bcf, bcl, cmap, bins, bin_step, blend);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static Color_Float_Array float_colormap(PyObject *py_colormap,
					bool require_contiguous,
					int nc, int size_divisor)
{
  Color_Array cmap = colormap(py_colormap, require_contiguous, nc, size_divisor);
  if (cmap.value_type() != Color_Array::Float)
    throw std::runtime_error("Colormap must have float values");
  return cmap;
}

// ----------------------------------------------------------------------------
//
static Color_Array colormap(PyObject *py_colormap, bool require_contiguous,
			    int nc, int size_divisor)
{
  bool allow_data_copy = false;
  Color_Array cmap = array_from_python(py_colormap, 2, allow_data_copy);
  if (nc > 0 && cmap.size(1) != nc)
    {
      std::ostringstream msg;
      msg << "The 2nd dimension of colormap array must have size "
	  << nc << ", got " << cmap.size(1);
      throw std::runtime_error(msg.str());
    }
  if (size_divisor > 0 && cmap.size(0) % size_divisor != 0)
    {
      std::ostringstream msg;
      msg << "Colormap size (" << cmap.size(0)
	<< ") must be a multiple of = "
	<< size_divisor;
      throw std::runtime_error(msg.str());
    }
  if (require_contiguous && !cmap.is_contiguous())
    throw std::runtime_error("Colormap array is non-contiguous");

  return cmap;
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
  float bcf1, bcl1, bcf2, bcl2;
  PyObject *py_colormap1, *py_colormap2;
  int bins = 0, bin_step = 1, iblend = 0;
  const char *kwlist[] = {"bcfirst1", "bclast1", "colormap1",
			  "bcfirst2", "bclast2", "colormap2",
			  "bins", "bin_step", "blend", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("ffOffO|iii"),
				   (char **)kwlist, &bcf1, &bcl1, &py_colormap1,
				   &bcf2, &bcl2, &py_colormap2,
				   &bins, &bin_step, &iblend))
    return NULL;
  bool blend = iblend;

  try
    {
      Color_Float_Array cmap1 = float_colormap(py_colormap1);
      Color_Float_Array cmap2 = float_colormap(py_colormap2, true, 0,
					       bins * bin_step);
      if (cmap2.size(1) != cmap1.size(1))
	throw std::runtime_error("Number of colors components doesn't match.");

      if (bins == 0)
	bins = cmap2.size(0);

      resample_colormap(bcf1, bcl1, cmap1, bcf2, bcl2, cmap2,
			bins, bin_step, blend);
    }
  catch (std::runtime_error &e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }

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
