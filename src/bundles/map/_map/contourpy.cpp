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
// Compute a contour surface.
//
#include <Python.h>			// use PyObject

// #include <iostream>			// use std:cerr for debugging

#include "contour.h"			// use surface()
#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use call_template_function()

using namespace Contour_Calculation;

// ----------------------------------------------------------------------------
//
template <class T>
void contour_surface(const Reference_Counted_Array::Array<T> &data,
		     float threshold, bool cap_faces, Contour_Surface **cs)
{
  // contouring calculation requires contiguous array
  // put sizes in x, y, z order
  AIndex size[3] = {static_cast<AIndex>(data.size(2)),
		    static_cast<AIndex>(data.size(1)),
		    static_cast<AIndex>(data.size(0))};
  GIndex stride[3] = {data.stride(2), data.stride(1), data.stride(0)};
  *cs = surface(data.values(), size, stride, threshold, cap_faces);
}

// ----------------------------------------------------------------------------
//
static PyObject *surface_py2(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_data;
  float threshold;
  int cap_faces = 1, return_normals = 0;
  const char *kwlist[] = {"data", "threshold", "cap_faces", "calculate_normals", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("Of|pp"),
				   (char **)kwlist,
				   &py_data, &threshold, &cap_faces,
				   &return_normals))
    return NULL;
  
  Numeric_Array data;
  if (!array_from_python(py_data, 3, &data))
    return NULL;

  Contour_Surface *cs;
  Py_BEGIN_ALLOW_THREADS
  call_template_function(contour_surface, data.value_type(),
  			 (data, threshold, cap_faces, &cs));
  Py_END_ALLOW_THREADS

  float *vxyz, *nxyz;
  VIndex *tvi;
  PyObject *vertex_xyz = python_float_array(cs->vertex_count(), 3, &vxyz);
  PyObject *normals = (return_normals ? python_float_array(cs->vertex_count(), 3, &nxyz) : NULL);
  PyObject *tv_indices = python_int_array(cs->triangle_count(), 3, &tvi);

  Py_BEGIN_ALLOW_THREADS
  cs->geometry(vxyz, reinterpret_cast<VIndex *>(tvi));
  if (return_normals)
      cs->normals(nxyz);

  delete cs;
  Py_END_ALLOW_THREADS

  PyObject *geom = (return_normals ?
		    python_tuple(vertex_xyz, tv_indices, normals) :
		    python_tuple(vertex_xyz, tv_indices));
  return geom;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *surface_py(PyObject *s, PyObject *args, PyObject *keywds)
{
  try
    {
      return surface_py2(s, args, keywds);
    }
  catch (std::bad_alloc&)
    {
      PyErr_SetString(PyExc_MemoryError, "Out of memory");
      return NULL;
    }
}

// ----------------------------------------------------------------------------
// Swap vertex 1 and 2 of each triangle.
//
static void reverse_triangle_vertex_order(IArray &triangles)
{
  VIndex *ta = triangles.values();
  TIndex n = triangles.size(0);
  TIndex s0 = triangles.stride(0), s1 = triangles.stride(1);
  for (TIndex t = 0 ; t < n ; ++t)
    {
      TIndex i1 = s0*t+s1, i2 = i1 + s1;
      VIndex v1 = ta[i1], v2 = ta[i2];
      ta[i1] = v2;
      ta[i2] = v1;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *reverse_triangle_vertex_order(PyObject *, PyObject *args)
{
  IArray tarray;
  if (!PyArg_ParseTuple(args, const_cast<char *>("O&"),
			parse_writable_int_n3_array, &tarray))
    return NULL;

  reverse_triangle_vertex_order(tarray);

  return python_none();
}
