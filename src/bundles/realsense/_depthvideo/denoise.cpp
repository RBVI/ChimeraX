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
// Pack triangles in Stereo Lithography (STL) file format.
//
#include <Python.h>			// use PyObject

//include <iostream>			// use std::cerr for debugging

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use FArray, IArray

static bool check_array_value_type(Numeric_Array &a, const char* name, Numeric_Array::Value_Type type);
static bool check_array_size(Numeric_Array &a, const char* name, int s0, int s1);
static bool check_array_size(Numeric_Array &a, const char* name, int s0, int s1, int s2);
static bool check_array_contiguous(Numeric_Array &a, const char* name);
static bool color_changed(unsigned char *c1, unsigned char *c2, int max_color_diff);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
denoise_depth(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array depth_image, color_image, ave_depth, max_depth, max_depth_color, last_color;
  float ave_weight;
  int max_color_diff;
  const char *kwlist[] = {"depth_image", "color_image", "ave_depth", "ave_weight",
			  "max_depth", "max_depth_color",
			  "last_color", "max_color_diff", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&fO&O&O&i"),
				   (char **)kwlist,
				   parse_2d_array, &depth_image,
				   parse_3d_array, &color_image,
				   parse_writable_2d_array, &ave_depth,
				   &ave_weight,
				   parse_writable_2d_array, &max_depth,
				   parse_writable_3d_array, &max_depth_color,
				   parse_writable_3d_array, &last_color,
				   &max_color_diff))
    return NULL;

  if (!check_array_value_type(depth_image, "depth_image", Numeric_Array::Unsigned_Short_Int) ||
      !check_array_value_type(color_image, "color_image", Numeric_Array::Unsigned_Char) ||
      !check_array_value_type(ave_depth, "ave_depth", Numeric_Array::Unsigned_Short_Int) ||
      !check_array_value_type(ave_depth, "max_depth", Numeric_Array::Unsigned_Short_Int) ||
      !check_array_value_type(last_color, "max_depth_color", Numeric_Array::Unsigned_Char) ||
      !check_array_value_type(last_color, "last_color", Numeric_Array::Unsigned_Char))
    return NULL;

  int s0 = depth_image.size(0), s1 = depth_image.size(1);
  if (!check_array_size(color_image, "color_image", s0, s1, 3) ||
      !check_array_size(ave_depth, "ave_depth", s0, s1) ||
      !check_array_size(ave_depth, "max_depth", s0, s1) ||
      !check_array_size(last_color, "max_depth_color", s0, s1, 3) ||
      !check_array_size(last_color, "last_color", s0, s1, 3))
    return NULL;

  if (!check_array_contiguous(depth_image, "depth_image") ||
      !check_array_contiguous(color_image, "color_image") ||
      !check_array_contiguous(ave_depth, "ave_depth") ||
      !check_array_contiguous(ave_depth, "max_depth") ||
      !check_array_contiguous(last_color, "max_depth_color") ||
      !check_array_contiguous(last_color, "last_color"))
    return NULL;

  unsigned short *ad = static_cast<unsigned short *>(ave_depth.values());
  unsigned short *md = static_cast<unsigned short *>(max_depth.values());
  unsigned short *d = static_cast<unsigned short *>(depth_image.values());
  unsigned char *c = static_cast<unsigned char *>(color_image.values());
  unsigned char *lc = static_cast<unsigned char *>(last_color.values());
  unsigned char *mdc = static_cast<unsigned char *>(max_depth_color.values());
  
  int i = 0, ci = 0;
  float f0 = ave_weight, f1 = 1-ave_weight;
  for (int i0 = 0 ; i0 < s0 ; ++i0)
    for (int i1 = 0 ; i1 < s1 ; ++i1, ++i, ci += 3)
      {
	unsigned short dv = d[i];
	if (dv >= md[i])
	  {
	    md[i] = dv;
	    for (int cc = 0 ; cc < 3 ; ++cc)
	      mdc[ci+cc] = c[ci+cc];
	  }
	if (dv == 0)
	  {
	    // Depth 0 indicates unknown depth
	    if (!color_changed(c+ci, mdc+ci, max_color_diff))
	      ad[i] = md[i];  // use max depth since color matches.
	  }
	else if (color_changed(c+ci, lc+ci, max_color_diff))
	  ad[i] = dv;	// Update depth immediately if color changed
	else
	  ad[i] = (unsigned short)(f0*dv + f1*ad[i]);  // Use average depth if color did not change
	for (int cc = 0 ; cc < 3 ; ++cc)
	  lc[cc+ci] = c[cc+ci];
      }
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
static bool color_changed(unsigned char *c1, unsigned char *c2, int max_color_diff)
{
  for (int cc = 0 ; cc < 3 ; ++cc)
    {
      int cdiff = (int)c1[cc] - (int)c2[cc];
      if (cdiff > max_color_diff || cdiff < -max_color_diff)
	return true;
    }
  return false;
}

// ----------------------------------------------------------------------------
//
static bool check_array_value_type(Numeric_Array &a, const char* name, Numeric_Array::Value_Type type)
{
  if (a.value_type() != type)
    {
      PyErr_Format(PyExc_TypeError, "%s array must have value type %s, got %s",
		   name, Numeric_Array::value_type_name(type),
		   Numeric_Array::value_type_name(a.value_type()));
      return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
static bool check_array_size(Numeric_Array &a, const char* name, int s0, int s1)
{
  if (a.size(0) != s0 || a.size(1) != s1)
    {
      PyErr_Format(PyExc_TypeError, "%s array size mismatch requires (%d, %d), got (%d, %d)",
		   name, s0, s1, a.size(0), a.size(1));
      return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
static bool check_array_size(Numeric_Array &a, const char* name, int s0, int s1, int s2)
{
  if (a.size(0) != s0 || a.size(1) != s1 || a.size(2) != s2)
    {
      PyErr_Format(PyExc_TypeError, "%s array size mismatch requires (%d, %d, %d), got (%d, %d, %d)",
		   name, s0, s1, s2, a.size(0), a.size(1), a.size(2));
      return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
static bool check_array_contiguous(Numeric_Array &a, const char* name)
{
  if (!a.is_contiguous())
    {
      PyErr_Format(PyExc_TypeError, "%s array must be contiguous", name);
      return false;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
static PyMethodDef depthvideo_methods[] = {
  {const_cast<char*>("denoise_depth"), (PyCFunction)denoise_depth,
   METH_VARARGS|METH_KEYWORDS,
   "denoise_depth(depth_image, color_image, ave_depth, ave_weight, max_depth, max_depth_color, last_color, max_color_diff)\n"
   "\n"
   "Denoise depth image by smoothing taking into account color changes.\n"
   "Implemented in C++.\n"
  },
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef depthvideo_def =
{
	PyModuleDef_HEAD_INIT,
	"_depthvideo",
	"Filtering for depth video",
	-1,
	depthvideo_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__depthvideo()
{
	return PyModule_Create(&depthvideo_def);
}
