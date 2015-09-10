// vi: set expandtab shiftwidth=4 softtabstop=4:
#include <Python.h>			// use PyObject

#include "blend_rgba.h"			// use blur_blend_images
#include "count.h"			// use count_values
#include "intercept.h"			// use closest_geometry_intercept
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

  /* intercept.h */
  {const_cast<char*>("closest_geometry_intercept"), (PyCFunction)closest_geometry_intercept,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("closest_sphere_intercept"), (PyCFunction)closest_sphere_intercept,
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
