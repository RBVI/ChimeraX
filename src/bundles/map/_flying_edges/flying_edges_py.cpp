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
// Compute a contour surface using Flying Edges algorithm.
//
#include <Python.h>			// use PyObject

// #include <iostream>			// use std:cerr for debugging

#include "flying_edges.h"		// use FlyingEdgesSurfaceWrapper
#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use call_template_function()

// ----------------------------------------------------------------------------
//
template <class T>
void flying_edges_surface(const Reference_Counted_Array::Array<T> &data,
                          float threshold, bool cap_faces, Contour_Surface **cs)
{
  // contouring calculation requires contiguous array
  // put sizes in x, y, z order
  AIndex size[3] = {static_cast<AIndex>(data.size(2)),
                    static_cast<AIndex>(data.size(1)),
                    static_cast<AIndex>(data.size(0))};
  GIndex stride[3] = {data.stride(2), data.stride(1), data.stride(0)};
  *cs = new FlyingEdges::FlyingEdgesSurfaceWrapper<T>(data.values(), size, stride,
                                                      threshold, cap_faces);
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
  call_template_function(flying_edges_surface, data.value_type(),
                         (data, threshold, cap_faces, &cs));
  Py_END_ALLOW_THREADS

  float *vxyz, *nxyz;
  VIndex *tvi;
  PyObject *vertex_xyz = python_float_array(cs->vertex_count(), 3, &vxyz);
  PyObject *normals = (return_normals ? python_float_array(cs->vertex_count(), 3, &nxyz) : NULL);
  PyObject *tv_indices = python_int_array(cs->triangle_count(), 3, &tvi);

  Py_BEGIN_ALLOW_THREADS
  if (return_normals)
      cs->geometry_with_normals(vxyz, reinterpret_cast<VIndex *>(tvi), nxyz);
  else
      cs->geometry(vxyz, reinterpret_cast<VIndex *>(tvi));

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
// Deferred API: create returns a handle + counts, generate writes into
// caller-provided pointers (e.g. VMA-mapped GPU memory), destroy frees.
// This avoids the numpy intermediate copy.
// ----------------------------------------------------------------------------

static PyObject *create_surface_py2(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_data;
  float threshold;
  int cap_faces = 1;
  const char *kwlist[] = {"data", "threshold", "cap_faces", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("Of|p"),
                                   (char **)kwlist,
                                   &py_data, &threshold, &cap_faces))
    return NULL;

  Numeric_Array data;
  if (!array_from_python(py_data, 3, &data))
    return NULL;

  Contour_Surface *cs;
  Py_BEGIN_ALLOW_THREADS
  call_template_function(flying_edges_surface, data.value_type(),
                         (data, threshold, cap_faces, &cs));
  Py_END_ALLOW_THREADS

  return Py_BuildValue("nKK",
      reinterpret_cast<Py_ssize_t>(cs),
      (unsigned long long)cs->vertex_count(),
      (unsigned long long)cs->triangle_count());
}

extern "C" PyObject *create_surface_py(PyObject *s, PyObject *args, PyObject *keywds)
{
  try { return create_surface_py2(s, args, keywds); }
  catch (std::bad_alloc&) {
    PyErr_SetString(PyExc_MemoryError, "Out of memory");
    return NULL;
  }
}

// Generate geometry into caller-provided raw pointers (e.g. mapped GPU
// memory).  vertex_ptr and triangle_ptr are Py_ssize_t addresses.
// normal_ptr may be 0 to skip normals.
static PyObject *generate_surface_py2(PyObject *, PyObject *args, PyObject *keywds)
{
  Py_ssize_t handle, vertex_ptr, triangle_ptr, normal_ptr = 0;
  const char *kwlist[] = {"handle", "vertex_ptr", "triangle_ptr",
                          "normal_ptr", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
          const_cast<char *>("nnn|n"), (char **)kwlist,
          &handle, &vertex_ptr, &triangle_ptr, &normal_ptr))
    return NULL;

  Contour_Surface *cs = reinterpret_cast<Contour_Surface *>(handle);
  float *vxyz = reinterpret_cast<float *>(vertex_ptr);
  VIndex *tvi = reinterpret_cast<VIndex *>(triangle_ptr);
  float *nxyz = normal_ptr ? reinterpret_cast<float *>(normal_ptr) : nullptr;

  Py_BEGIN_ALLOW_THREADS
  if (nxyz)
    cs->geometry_with_normals(vxyz, tvi, nxyz);
  else
    cs->geometry(vxyz, tvi);
  Py_END_ALLOW_THREADS

  Py_RETURN_NONE;
}

extern "C" PyObject *generate_surface_py(PyObject *s, PyObject *args, PyObject *keywds)
{
  try { return generate_surface_py2(s, args, keywds); }
  catch (std::bad_alloc&) {
    PyErr_SetString(PyExc_MemoryError, "Out of memory");
    return NULL;
  }
}

extern "C" PyObject *destroy_surface_py(PyObject *, PyObject *args)
{
  Py_ssize_t handle;
  if (!PyArg_ParseTuple(args, const_cast<char *>("n"), &handle))
    return NULL;
  delete reinterpret_cast<Contour_Surface *>(handle);
  Py_RETURN_NONE;
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
