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

#include <iostream>			// use std::cerr for debugging
#include <Python.h>			// use PyObject

#include "bounds.h"			// use sphere_bounds, ...
#include "closepoints.h"		// use find_close_points, ...
#include "cylinderrot.h"		// use cylinder_rotations
#include "distancespy.h"		// use py_distances_from_origin, ...
#include "intercept.h"			// use closest_geometry_intercept
#include "spline.h"			// use natural_cubic_spline
#include "transform.h"			// use affine_transform_vertices, ...
#include "vector_ops.h"			// use inner_product_64
#include "matrix.h"		        // defines look_at
#include "fill_ring.h"		        // defines fill_small_ring and fill_6ring

namespace Geometry_Cpp
{

// ----------------------------------------------------------------------------
//
static struct PyMethodDef geometry_cpp_methods[] =
{

  /* bounds.h */
  {const_cast<char*>("point_bounds"), (PyCFunction)point_bounds, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("point_copies_bounds"), (PyCFunction)point_copies_bounds, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("sphere_bounds"), (PyCFunction)sphere_bounds, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("sphere_axes_bounds"), (PyCFunction)sphere_axes_bounds, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("spheres_in_bounds"), (PyCFunction)spheres_in_bounds, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("bounds_overlap"), (PyCFunction)bounds_overlap, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("points_within_planes"), (PyCFunction)points_within_planes, METH_VARARGS|METH_KEYWORDS, NULL},

  /* closepoints.h */
  {const_cast<char*>("find_close_points"), (PyCFunction)find_close_points,
   METH_VARARGS|METH_KEYWORDS, find_close_points_doc},
  {const_cast<char*>("find_closest_points"), (PyCFunction)find_closest_points,
   METH_VARARGS|METH_KEYWORDS, find_closest_points_doc},
  {const_cast<char*>("find_close_points_sets"), (PyCFunction)find_close_points_sets,
   METH_VARARGS|METH_KEYWORDS, find_close_points_sets_doc},

  /* cylinderrot.h */
  {const_cast<char*>("cylinder_rotations"), (PyCFunction)cylinder_rotations, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("half_cylinder_rotations"), (PyCFunction)half_cylinder_rotations, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("cylinder_rotations_x3d"), (PyCFunction)cylinder_rotations_x3d, METH_VARARGS|METH_KEYWORDS, NULL},

  /* distancepy.h */
  {const_cast<char*>("distances_from_origin"), py_distances_from_origin, METH_VARARGS, NULL},
  {const_cast<char*>("distances_perpendicular_to_axis"), py_distances_perpendicular_to_axis, METH_VARARGS, NULL},
  {const_cast<char*>("distances_parallel_to_axis"), py_distances_parallel_to_axis, METH_VARARGS, NULL},
  {const_cast<char*>("maximum_norm"), (PyCFunction)py_maximum_norm, METH_VARARGS|METH_KEYWORDS, NULL},

  /* intercept.h */
  {const_cast<char*>("closest_triangle_intercept"), (PyCFunction)closest_triangle_intercept,
   METH_VARARGS|METH_KEYWORDS, closest_triangle_intercept_doc},
  {const_cast<char*>("closest_sphere_intercept"), (PyCFunction)closest_sphere_intercept,
   METH_VARARGS|METH_KEYWORDS, closest_sphere_intercept_doc},
  {const_cast<char*>("segment_intercepts_spheres"), (PyCFunction)segment_intercepts_spheres,
   METH_VARARGS|METH_KEYWORDS, segment_intercepts_spheres_doc},
  {const_cast<char*>("closest_cylinder_intercept"), (PyCFunction)closest_cylinder_intercept,
   METH_VARARGS|METH_KEYWORDS, closest_cylinder_intercept_doc},

  /* matrix.h */
  {const_cast<char*>("look_at"), (PyCFunction)look_at, METH_VARARGS, NULL},
  {const_cast<char*>("multiply_matrices"), (PyCFunction)multiply_matrices,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("multiply_matrix_lists"), (PyCFunction)multiply_matrix_lists,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("same_matrix"), (PyCFunction)same_matrix,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("set_translation_matrix"), (PyCFunction)set_translation_matrix,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("set_scale_matrix"), (PyCFunction)set_scale_matrix,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("is_identity_matrix"), (PyCFunction)is_identity_matrix,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("opengl_matrix"), (PyCFunction)opengl_matrix,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("opengl_matrices"), (PyCFunction)opengl_matrices,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("invert_orthonormal"), (PyCFunction)invert_orthonormal,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* spline.h */
  {const_cast<char*>("natural_cubic_spline"), (PyCFunction)natural_cubic_spline,
   METH_VARARGS|METH_KEYWORDS, natural_cubic_spline_doc},

  /* transform.h */
  {const_cast<char*>("scale_and_shift_vertices"), scale_and_shift_vertices, METH_VARARGS, NULL},
  {const_cast<char*>("scale_vertices"), scale_vertices, METH_VARARGS, NULL},
  {const_cast<char*>("shift_vertices"), shift_vertices, METH_VARARGS, NULL},
  {const_cast<char*>("affine_transform_vertices"), affine_transform_vertices, METH_VARARGS, NULL},
  {const_cast<char*>("affine_transform_normals"), affine_transform_normals, METH_VARARGS, NULL},

  /* vector_ops.h */
  {const_cast<char*>("inner_product_64"), (PyCFunction)inner_product_64,
   METH_VARARGS|METH_KEYWORDS, inner_product_64_doc},

  /* fill_ring.h */
  // defines fill_small_ring and fill_6ring
  {const_cast<char*>("fill_small_ring"), (PyCFunction)fill_small_ring,
   METH_VARARGS|METH_KEYWORDS, fill_small_ring_doc},
  {const_cast<char*>("fill_6ring"), (PyCFunction)fill_6ring,
   METH_VARARGS|METH_KEYWORDS, fill_6ring_doc},

  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((Geometry_Cpp::module_state*)PyModule_GetState(m))

static int geometry_cpp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int geometry_cpp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_geometry",
        NULL,
        sizeof(struct module_state),
        geometry_cpp_methods,
        NULL,
        geometry_cpp_traverse,
        geometry_cpp_clear,
        NULL
};

}	// Geometry_Cpp namespace

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
PyMODINIT_FUNC
PyInit__geometry(void)
{
    PyObject *module = PyModule_Create(&Geometry_Cpp::moduledef);
    
    if (module == NULL)
      return NULL;
    Geometry_Cpp::module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_geometry.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
