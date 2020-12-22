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

#include <Python.h>			// use PyObject

#include "normals.h"			// use parallel_transport
#include "spline.h"			// use cubic_path
#include "xsection.h"			// use rxsection_*

// ----------------------------------------------------------------------------
//
static struct PyMethodDef ribbons_cpp_methods[] =
{
  /* normals.h */
  {const_cast<char*>("parallel_transport"), (PyCFunction)parallel_transport_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("smooth_twist"), (PyCFunction)smooth_twist_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("dihedral_angle"), (PyCFunction)dihedral_angle_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("path_plane_normals"), (PyCFunction)path_plane_normals,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("path_guide_normals"), (PyCFunction)path_guide_normals,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* spline.h */
  {const_cast<char*>("cubic_path"), (PyCFunction)cubic_path,
   METH_VARARGS|METH_KEYWORDS, cubic_path_doc},
  {const_cast<char*>("spline_path"), (PyCFunction)spline_path,
   METH_VARARGS|METH_KEYWORDS, spline_path_doc},
  {const_cast<char*>("cubic_spline"), (PyCFunction)cubic_spline,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("atom_spline_positions"), (PyCFunction)atom_spline_positions,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("set_atom_tether_positions"), (PyCFunction)set_atom_tether_positions,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("get_polymer_spline"), (PyCFunction)get_polymer_spline,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* xsection.h */
  {const_cast<char*>("rxsection_new"), (PyCFunction)rxsection_new,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("rxsection_delete"), (PyCFunction)rxsection_delete,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("rxsection_extrude"), (PyCFunction)rxsection_extrude,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("rxsection_scale"), (PyCFunction)rxsection_scale,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("rxsection_arrow"), (PyCFunction)rxsection_arrow,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("ribbon_extrusions"), (PyCFunction)ribbon_extrusions,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("geometry_new"), (PyCFunction)geometry_new,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("geometry_delete"), (PyCFunction)geometry_delete,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("geometry_add_range"), (PyCFunction)geometry_add_range,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("geometry_set_range_offset"), (PyCFunction)geometry_set_range_offset,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("geometry_ranges"), (PyCFunction)geometry_ranges,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("geometry_empty"), (PyCFunction)geometry_empty,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("geometry_arrays"), (PyCFunction)geometry_arrays,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("ribbon_vertex_colors"), (PyCFunction)ribbon_vertex_colors,
   METH_VARARGS|METH_KEYWORDS, NULL},

  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((module_state*)PyModule_GetState(m))

static int ribbons_cpp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int ribbons_cpp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_ribbons",
        NULL,
        sizeof(struct module_state),
        ribbons_cpp_methods,
        NULL,
        ribbons_cpp_traverse,
        ribbons_cpp_clear,
        NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
PyMODINIT_FUNC
PyInit__ribbons(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    
    if (module == NULL)
      return NULL;
    module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_ribbons.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
