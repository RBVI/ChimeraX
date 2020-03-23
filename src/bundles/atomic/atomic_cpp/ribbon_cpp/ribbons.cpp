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

#include "normals.h"			// use ribbon_constrained_normals

// ----------------------------------------------------------------------------
//
static struct PyMethodDef ribbons_cpp_methods[] =
{
  /* normals.h */
  {const_cast<char*>("parallel_transport"), (PyCFunction)parallel_transport,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("smooth_twist"), (PyCFunction)smooth_twist,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("dihedral_angle"), (PyCFunction)dihedral_angle,
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
