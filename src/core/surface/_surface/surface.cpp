// vi: set expandtab shiftwidth=4 softtabstop=4:
#include <iostream>			// use std::cerr for debugging
#include <Python.h>			// use PyObject

#include "connected.h"			// use connected_triangles, ...
#include "measure.h"			// use enclosed_volume, surface_area, ...
#include "normals.h"			// use calculate_vertex_normals, invert_vertex_normals
#include "parse_stl.h"			// use parse_stl
#include "patches.h"			// use sharp_edge_patches
#include "sasa.h"			// use surface_area_of_spheres
#include "subdivide.h"			// use subdivide_triangles
#include "tube.h"			// use tube_geometry

namespace Surface_Cpp
{

// ----------------------------------------------------------------------------
//
static struct PyMethodDef surface_cpp_methods[] =
{
  /* connected.h */
  {const_cast<char*>("connected_triangles"), (PyCFunction)connected_triangles,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("triangle_vertices"), (PyCFunction)triangle_vertices,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("connected_pieces"), (PyCFunction)connected_pieces,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* measure.h */
  {const_cast<char*>("enclosed_volume"), (PyCFunction)enclosed_volume,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("surface_area"), (PyCFunction)surface_area,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("vertex_areas"), (PyCFunction)vertex_areas,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("boundary_edges"), (PyCFunction)boundary_edges,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("boundary_loops"), (PyCFunction)boundary_loops,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* normals.h */
  {const_cast<char*>("calculate_vertex_normals"), (PyCFunction)calculate_vertex_normals,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("invert_vertex_normals"), (PyCFunction)invert_vertex_normals,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* parse_stl.h */
  {const_cast<char*>("parse_stl"), (PyCFunction)parse_stl,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* patches.h */
  {const_cast<char*>("sharp_edge_patches"), (PyCFunction)sharp_edge_patches,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* sasa.h */
  {const_cast<char*>("surface_area_of_spheres"), (PyCFunction)surface_area_of_spheres,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("estimate_surface_area_of_spheres"), (PyCFunction)estimate_surface_area_of_spheres,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* subdivide.h */
  {const_cast<char*>("subdivide_triangles"), (PyCFunction)subdivide_triangles,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("subdivide_mesh"), (PyCFunction)subdivide_mesh,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* tube.h */
  {const_cast<char*>("tube_geometry"), (PyCFunction)tube_geometry,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("tube_geometry_colors"), (PyCFunction)tube_geometry_colors,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("tube_triangle_mask"), (PyCFunction)tube_triangle_mask,
   METH_VARARGS|METH_KEYWORDS, NULL},

  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int surface_cpp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int surface_cpp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "surface_cpp",
        NULL,
        sizeof(struct module_state),
        surface_cpp_methods,
        NULL,
        surface_cpp_traverse,
        surface_cpp_clear,
        NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
extern "C" PyObject *
PyInit__surface(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    
    if (module == NULL)
      return NULL;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("surface_cpp.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}

}	// Surface_Cpp namespace
