// vi: set expandtab shiftwidth=4 softtabstop=4:

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
// Pack triangles in Stereo Lithography (STL) file format.
//
#include <Python.h>			// use PyObject

//include <iostream>			// use std::cerr for debugging

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use Numeric_Array

static bool check_rgba_array(Numeric_Array &rgba);

// ----------------------------------------------------------------------------
//
inline void y2uv_to_rgba(float y0, float y1, float u, float v, unsigned int *rgba)
{
  // From https://www.fourcc.org/fccyvrgb.php
  // B = 1.164(Y - 16)                   + 2.018(U - 128)
  // G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
  // R = 1.164(Y - 16) + 1.596(V - 128)
  float y0s = 1.164f * (y0 - 16.f), y1s = 1.164f * (y1 - 16.f);
  float us = (u - 128.f), vs = (v - 128.f);

  float b0 = y0s               + 2.018f * us;
  float g0 = y0s - 0.813f * vs - 0.391f * us;
  float r0 = y0s + 1.596f * vs;
  unsigned int ir0 = (r0 < 0 ? 0 : (r0 >= 256 ? 255 : (int)r0));
  unsigned int ig0 = (g0 < 0 ? 0 : (g0 >= 256 ? 255 : (int)g0));
  unsigned int ib0 = (b0 < 0 ? 0 : (b0 >= 256 ? 255 : (int)b0));
  unsigned int a = 0xff000000;
  *rgba = ir0 | (ig0 << 8) | (ib0 << 16) | a;
  
  float b1 = y1s              + 2.018f * us;
  float g1 = y1s - 0.813f * vs - 0.391f * us;
  float r1 = y1s + 1.596f * vs;
  unsigned int ir1 = (r1 < 0 ? 0 : (r1 >= 256 ? 255 : (int)r1));
  unsigned int ig1 = (g1 < 0 ? 0 : (g1 >= 256 ? 255 : (int)g1));
  unsigned int ib1 = (b1 < 0 ? 0 : (b1 >= 256 ? 255 : (int)b1));
  rgba[1] = ir1 | (ig1 << 8) | (ib1 << 16) | a;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
bgra_to_rgba(PyObject *, PyObject *args, PyObject *keywds)
{
  void *bgra_data;
  Numeric_Array rgba_image;
  int padded_width;
  const char *kwlist[] = {"bgra_data", "rgba_array", "padded_width", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&i"),
				   (char **)kwlist,
				   parse_voidp, &bgra_data,
				   parse_writable_3d_array, &rgba_image,
				   &padded_width))
    return NULL;

  if (!check_rgba_array(rgba_image))
    return NULL;

  unsigned int *bgra = static_cast<unsigned int *>(bgra_data);
  unsigned int *rgba = static_cast<unsigned int *>(rgba_image.values());
  int64_t h = rgba_image.size(0), w = rgba_image.size(1);
  for (int64_t r = 0 ; r < h ; ++r)
    for (int64_t c = 0 ; c < w ; ++c)
    {
      int64_t i = r*padded_width + c;
      unsigned int p = bgra[i];
      // Swap red and blue.
      rgba[r*w+c] = ((p & 0xff0000) >> 16) | (p & 0xff00) | ((p & 0xff) << 16) | (p & 0xff000000);
    }
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
yuyv_to_rgba(PyObject *, PyObject *args, PyObject *keywds)
{
  void *yuyv_data;
  Numeric_Array rgba_image;
  int padded_width;
  const char *kwlist[] = {"yuyv_data", "rgba_array", "padded_width", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&i"),
				   (char **)kwlist,
				   parse_voidp, &yuyv_data,
				   parse_writable_3d_array, &rgba_image,
				   &padded_width))
    return NULL;

  if (!check_rgba_array(rgba_image))
    return NULL;

  unsigned int *yuyv = static_cast<unsigned int *>(yuyv_data);
  unsigned int *rgba = static_cast<unsigned int *>(rgba_image.values());
  int64_t h = rgba_image.size(0), w = rgba_image.size(1);
  int64_t w2 = w/2, pw2 = padded_width/2;
  for (int64_t r = 0 ; r < h ; ++r)
    for (int64_t c = 0 ; c < w2 ; ++c)
    {
      int64_t i = r*pw2 + c;
      unsigned int p = yuyv[i];
      float y0 = (float)(p & 0xff);
      float u = (float)((p & 0xff00) >> 8);
      float y1 = (float)((p & 0xff0000) >> 16);
      float v = (float)((p & 0xff000000) >> 24);
      
      y2uv_to_rgba(y0,y1,u,v,rgba+(r*w+2*c));
    }
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
uyvy_to_rgba(PyObject *, PyObject *args, PyObject *keywds)
{
  void *uyvy_data;
  Numeric_Array rgba_image;
  int padded_width;
  const char *kwlist[] = {"uyvy_data", "rgba_array", "padded_width", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&i"),
				   (char **)kwlist,
				   parse_voidp, &uyvy_data,
				   parse_writable_3d_array, &rgba_image,
				   &padded_width))
    return NULL;

  if (!check_rgba_array(rgba_image))
    return NULL;

  unsigned int *uyvy = static_cast<unsigned int *>(uyvy_data);
  unsigned int *rgba = static_cast<unsigned int *>(rgba_image.values());
  int64_t h = rgba_image.size(0), w = rgba_image.size(1);
  int64_t w2 = w/2, pw2 = padded_width/2;
  for (int64_t r = 0 ; r < h ; ++r)
    for (int64_t c = 0 ; c < w2 ; ++c)
    {
      int64_t i = r*pw2 + c;
      unsigned int p = uyvy[i];
      float u = (float)(p & 0xff);
      float y0 = (float)((p & 0xff00) >> 8);
      float v = (float)((p & 0xff0000) >> 16);
      float y1 = (float)((p & 0xff000000) >> 24);

      y2uv_to_rgba(y0,y1,u,v,rgba+(r*w+2*c));
    }
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
nv12_to_rgba(PyObject *, PyObject *args, PyObject *keywds)
{
  void *y_data, *uv_data;
  Numeric_Array rgba_image;
  int padded_width;
  const char *kwlist[] = {"y_data", "uv_data", "rgba_array", "padded_width", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&i"),
				   (char **)kwlist,
				   parse_voidp, &y_data,
				   parse_voidp, &uv_data,
				   parse_writable_3d_array, &rgba_image,
				   &padded_width))
    return NULL;

  if (!check_rgba_array(rgba_image))
    return NULL;

  int64_t h = rgba_image.size(0), w = rgba_image.size(1);
  unsigned short int *y0 = static_cast<unsigned short int *>(y_data);
  unsigned short int *y1 = y0 + padded_width/2;
  unsigned short int *uv = static_cast<unsigned short int *>(uv_data);
  unsigned int *rgba0 = static_cast<unsigned int *>(rgba_image.values());
  unsigned int *rgba1 = rgba0 + w;
  int64_t w2 = padded_width/2, w4 = padded_width/4, iw2 = w/2;
  for (int64_t r = 0 ; r < h ; r += 2)
    for (int64_t c = 0 ; c < iw2 ; ++c)
    {
      int64_t i = r*w2 + c;
      unsigned short int y0i = y0[i];
      float y00 = (float)(y0i & 0xff);
      float y01 = (float)((y0i & 0xff00) >> 8);
      unsigned short int uvi = uv[r*w4+c];
      float u = (float)(uvi & 0xff);
      float v = (float)((uvi & 0xff00) >> 8);
      int64_t j = r*w+2*c;
      y2uv_to_rgba(y00,y01,u,v,rgba0+j);

      unsigned short int y1i = y1[i];
      float y10 = (float)(y1i & 0xff);
      float y11 = (float)((y1i & 0xff00) >> 8);
      y2uv_to_rgba(y10,y11,u,v,rgba1+j);
    }
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
inline bool color_match(unsigned int rgba, int *comps, int nc, int saturation)
{
  for (int i = 0 ; i < nc ; ++i)
    {
      unsigned int c1 = (rgba >> 8*comps[2*i]) & 0xff;
      unsigned int c2 = (rgba >> 8*comps[2*i+1]) & 0xff;
      if (c1 < c2 + saturation)
	return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
set_color_alpha(PyObject *, PyObject *args, PyObject *keywds)
{
  BArray color;
  Numeric_Array rgba_image;
  int alpha, saturation;
  const char *kwlist[] = {"rgba_array", "color", "saturation", "alpha", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&ii"),
				   (char **)kwlist,
				   parse_writable_3d_array, &rgba_image,
				   parse_uint8_n_array, &color,
				   &saturation,
				   &alpha))
    return NULL;

  if (!check_rgba_array(rgba_image))
    return NULL;
  if (color.size(0) != 4)
    {
      PyErr_Format(PyExc_TypeError, "color had size %s, require 4",
		   color.size_string().c_str());
      return NULL;
    }

  // Determine which pairs of color components to compare
  // based on relative size of components of specified color.
  unsigned char *ca = color.values();
  int comps[6], nc = 0;
  for (int i = 0 ; i < 3 ; ++i)
    {
      int i1 = (i+1)%3;
      if (ca[i] > ca[i1])
	{ comps[2*nc] = i; comps[2*nc+1] = i1; nc += 1; }
      else if (ca[i] < ca[i1])
	{ comps[2*nc] = i1; comps[2*nc+1] = i; nc += 1; }
    }

  // Set alpha value for image pixels that have color with
  // same components of similar relative size.
  unsigned int *rgba = static_cast<unsigned int *>(rgba_image.values());
  int64_t h = rgba_image.size(0), w = rgba_image.size(1);
  int64_t wh = w*h;
  unsigned int a24 = ((unsigned int)alpha) << 24;
  for (int64_t i = 0 ; i < wh ; ++i)
    {
      unsigned int c = rgba[i];
      if (color_match(c, comps, nc, saturation))
	  rgba[i] = (c & 0xffffff) | a24;
    }
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
static bool check_rgba_array(Numeric_Array &rgba)
{
  if (rgba.size(2) != 4)
    {
      PyErr_Format(PyExc_TypeError, "rgba_image array third dimension must have size 4, got %s",
		   rgba.size_string().c_str());
      return false;
    }
  if (!rgba.is_contiguous())
    {
      PyErr_Format(PyExc_TypeError, "rgba_image array must be continguous");
      return false;
    }
  if (rgba.value_type() != Numeric_Array::Unsigned_Char)
    {
      PyErr_Format(PyExc_TypeError, "rgba_image array type must be unsigned char, got %s",
		   rgba.value_type_name(rgba.value_type()));
      return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
static PyMethodDef webcam_methods[] = {
  {const_cast<char*>("bgra_to_rgba"), (PyCFunction)bgra_to_rgba,
   METH_VARARGS|METH_KEYWORDS,
   "bgra_to_rgba(bgra_data, rgba_array, padded_width)\n"
   "\n"
   "Convert bgra pixels to rgba pixels for a 2D array.\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("yuyv_to_rgba"), (PyCFunction)yuyv_to_rgba,
   METH_VARARGS|METH_KEYWORDS,
   "yuyv_to_rgba(yuyv_data, rgba_array, padded_width)\n"
   "\n"
   "Convert yuyv pixels to rgba pixels for a 2D array.\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("uyvy_to_rgba"), (PyCFunction)uyvy_to_rgba,
   METH_VARARGS|METH_KEYWORDS,
   "uyvy_to_rgba(uyvy_data, rgba_array, padded_width)\n"
   "\n"
   "Convert uyvy pixels to rgba pixels for a 2D array.\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("nv12_to_rgba"), (PyCFunction)nv12_to_rgba,
   METH_VARARGS|METH_KEYWORDS,
   "nv12_to_rgba(y_data, uv_data, rgba_array, padded_width)\n"
   "\n"
   "Convert nv12 pixels to rgba pixels for a 2D array.\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("set_color_alpha"), (PyCFunction)set_color_alpha,
   METH_VARARGS|METH_KEYWORDS,
   "set_color_alpha(rgba_array, color, saturation, alpha)\n"
   "\n"
   "Set image alpha values for colors that match the specified color.\n"
   "Color matching is based on relative color component sizes and\n"
   "the saturation value.\n"
   "Implemented in C++.\n"
  },
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef webcam_def =
{
	PyModuleDef_HEAD_INIT,
	"_webcam",
	"Pixel conversions for web camera video",
	-1,
	webcam_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__webcam()
{
	return PyModule_Create(&webcam_def);
}
