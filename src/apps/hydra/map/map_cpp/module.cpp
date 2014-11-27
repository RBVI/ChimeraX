#include <iostream>			// use std::cerr for debugging
#include <Python.h>			// use PyObject

#include "combine.h"			// use linear_combination
#include "contourpy.h"			// use surface_py, ...
#include "fittingpy.h"			// use py_correlation_gradient, ...
#include "distgrid.h"			// use py_sphere_surface_distance
#include "gaussian.h"			// use py_sum_of_gaussians
#include "histogram.h"			// use bin_counts_py, ...
#include "interpolatepy.h"		// use interpolate_volume_data, ...
#include "occupancy.h"			// use fill_occupancy_map
#include "squaremesh.h"			// use principle_plane_edges
#include "transfer.h"			// use data_to_rgba,...

namespace Map_Cpp
{

// ----------------------------------------------------------------------------
//
static struct PyMethodDef map_cpp_methods[] =
{
  /* combine.h */
  {const_cast<char*>("linear_combination"), (PyCFunction)linear_combination,
   METH_VARARGS|METH_KEYWORDS},

  /* contourpy.h */
  {const_cast<char*>("contour_surface"), (PyCFunction)surface_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("reverse_triangle_vertex_order"),
   reverse_triangle_vertex_order, METH_VARARGS, NULL},

  /* distgrid.h */
  {const_cast<char*>("sphere_surface_distance"), (PyCFunction)py_sphere_surface_distance,
   METH_VARARGS|METH_KEYWORDS},

  /* fittingpy.h */
  {const_cast<char*>("correlation_gradient"), (PyCFunction)py_correlation_gradient, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("torque"), (PyCFunction)py_torque, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("torques"), (PyCFunction)py_torques, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("correlation_torque"), (PyCFunction)py_correlation_torque, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("correlation_torque2"), (PyCFunction)py_correlation_torque2, METH_VARARGS|METH_KEYWORDS, NULL},

  /* gaussian.h */
  {const_cast<char*>("sum_of_gaussians"), (PyCFunction)py_sum_of_gaussians,
   METH_VARARGS|METH_KEYWORDS},
  {const_cast<char*>("sum_of_balls"), (PyCFunction)py_sum_of_balls,
   METH_VARARGS|METH_KEYWORDS},

  /* histogram.h */
  {const_cast<char*>("minimum_and_maximum"), (PyCFunction)minimum_and_maximum,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("bin_counts"), (PyCFunction)bin_counts_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("high_count"), (PyCFunction)high_count_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("high_indices"), (PyCFunction)high_indices_py,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* interpolatepy.h */
  {const_cast<char*>("interpolate_volume_data"), interpolate_volume_data, METH_VARARGS, NULL},
  {const_cast<char*>("interpolate_volume_gradient"), interpolate_volume_gradient, METH_VARARGS, NULL},
  {const_cast<char*>("interpolate_colormap"), interpolate_colormap, METH_VARARGS, NULL},
  {const_cast<char*>("set_outside_volume_colors"), set_outside_volume_colors, METH_VARARGS, NULL},

  /* occupancy.h */
  {const_cast<char*>("fill_occupancy_map"), (PyCFunction)fill_occupancy_map,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* squaremesh.h */
  {const_cast<char*>("principle_plane_edges"), (PyCFunction)principle_plane_edges,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* transfer.h */
  {const_cast<char*>("transfer_function_colormap"), (PyCFunction)transfer_function_colormap,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("colors_float_to_uint"), (PyCFunction)colors_float_to_uint,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("data_to_colors"), (PyCFunction)data_to_colors,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("data_to_colormap_colors"), (PyCFunction)data_to_colormap_colors,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("data_to_bin_index"), (PyCFunction)data_to_bin_index,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("indices_to_colors"), (PyCFunction)indices_to_colors,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("resample_colormap"), (PyCFunction)resample_colormap,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("data_to_rgba"), (PyCFunction)data_to_rgba,
   METH_VARARGS|METH_KEYWORDS, NULL},

  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int map_cpp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int map_cpp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "map_cpp",
        NULL,
        sizeof(struct module_state),
        map_cpp_methods,
        NULL,
        map_cpp_traverse,
        map_cpp_clear,
        NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
extern "C" PyObject *
PyInit_map_cpp(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    
    if (module == NULL)
      return NULL;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("map_cpp.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}

}	// Map_Cpp namespace
