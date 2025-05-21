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

#include <Python.h>			// use PyObject

#include "blend_rgba.h"			// use blur_blend_images
#include "count.h"			// use count_values
#include "linux_swap_interval.h"	// use set_linux_swap_interval
#include "mac_swap_interval.h"		// use set_mac_swap_interval
#include "mesh_edges.h"			// use masked_edges

namespace Graphics_Cpp
{

// ----------------------------------------------------------------------------
//
static struct PyMethodDef graphics_cpp_methods[] =
{
  /* blend_rgba.h */
  {const_cast<char*>("blur_blend_images"), (PyCFunction)blur_blend_images,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("accumulate_images"), (PyCFunction)accumulate_images,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* count.h */
  {const_cast<char*>("count_value"), (PyCFunction)count_value,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* linux_swap_interval.h */
  {const_cast<char*>("set_linux_swap_interval"), (PyCFunction)set_linux_swap_interval,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* mac_swap_interval.h */
  {const_cast<char*>("set_mac_swap_interval"), (PyCFunction)set_mac_swap_interval,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* mesh_edges.h */
  {const_cast<char*>("masked_edges"), (PyCFunction)masked_edges,
   METH_VARARGS|METH_KEYWORDS, NULL},

  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((Graphics_Cpp::module_state*)PyModule_GetState(m))

static int graphics_cpp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int graphics_cpp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_graphics",
        NULL,
        sizeof(struct module_state),
        graphics_cpp_methods,
        NULL,
        graphics_cpp_traverse,
        graphics_cpp_clear,
        NULL
};

}	// Graphics_Cpp namespace

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
PyMODINIT_FUNC
PyInit__graphics(void)
{
    PyObject *module = PyModule_Create(&Graphics_Cpp::moduledef);
    
    if (module == NULL)
      return NULL;
    Graphics_Cpp::module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_graphics.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
