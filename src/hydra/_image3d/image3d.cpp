#include <iostream>			// use std::cerr for debugging
#include <Python.h>			// use PyObject

#include "blend_rgba.h"			// use blur_blend_images
#include "combine.h"			// use linear_combination
#include "gaussian.h"			// use py_sum_of_gaussians
#include "histogram.h"			// use bin_counts_py, ...
#include "intercept.h"			// use closest_geometry_intercept
#include "mesh_edges.h"			// use masked_edges
#include "parsecif.h"			// use parse_mmcif_file
#include "parsepdb.h"			// use parse_pdb_file
#include "parse_stl.h"			// use parse_stl
#include "pdb_bonds.h"			// use molecule_bonds
#include "pycontour.h"			// use surface_py, ...
#include "pydistances.h"		// use py_distances_from_origin, ...
#include "pyinterpolate.h"		// use interpolate_volume_data, ...
#include "spline.h"			// use natural_cubic_spline
#include "squaremesh.h"			// use principle_plane_edges
#include "subdivide.h"			// use subdivide_triangles
#include "transfer.h"			// use data_to_rgba,...
#include "tube.h"			// use tube_geometry

namespace Image_3d
{

// ----------------------------------------------------------------------------
//
static struct PyMethodDef image3d_methods[] =
{
  /* blend_rgba.h */
  {const_cast<char*>("blur_blend_images"), (PyCFunction)blur_blend_images,
   METH_VARARGS|METH_KEYWORDS},

  /* combine.h */
  {const_cast<char*>("linear_combination"), (PyCFunction)linear_combination,
   METH_VARARGS|METH_KEYWORDS},
  {const_cast<char*>("inner_product_64"), (PyCFunction)inner_product_64,
   METH_VARARGS|METH_KEYWORDS},

  /* gaussian.h */
  {const_cast<char*>("sum_of_gaussians"), (PyCFunction)py_sum_of_gaussians,
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

  /* intercept.h */
  {const_cast<char*>("closest_geometry_intercept"), (PyCFunction)closest_geometry_intercept,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("closest_sphere_intercept"), (PyCFunction)closest_sphere_intercept,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* parsecif.h */
  {const_cast<char*>("parse_mmcif_file"), (PyCFunction)parse_mmcif_file,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* parsepdb.h */
  {const_cast<char*>("parse_pdb_file"), (PyCFunction)parse_pdb_file,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("element_radii"), (PyCFunction)element_radii,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* parse_stl.h */
  {const_cast<char*>("parse_stl"), (PyCFunction)parse_stl,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* pdb_bonds.h */
  {const_cast<char*>("molecule_bonds"), (PyCFunction)molecule_bonds,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("initialize_bond_templates"), (PyCFunction)initialize_bond_templates,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* pydistances.h */
  {const_cast<char*>("distances_from_origin"), py_distances_from_origin,	METH_VARARGS, NULL},
  {const_cast<char*>("distances_perpendicular_to_axis"), py_distances_perpendicular_to_axis,	METH_VARARGS, NULL},
  {const_cast<char*>("distances_parallel_to_axis"), py_distances_parallel_to_axis,	METH_VARARGS, NULL},
  {const_cast<char*>("maximum_norm"), (PyCFunction)py_maximum_norm, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("correlation_gradient"), (PyCFunction)py_correlation_gradient, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("torque"), (PyCFunction)py_torque, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("torques"), (PyCFunction)py_torques, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("correlation_torque"), (PyCFunction)py_correlation_torque, METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("correlation_torque2"), (PyCFunction)py_correlation_torque2, METH_VARARGS|METH_KEYWORDS, NULL},

  /* pyinterpolate.h */
  {const_cast<char*>("interpolate_volume_data"), interpolate_volume_data, METH_VARARGS, NULL},
  {const_cast<char*>("interpolate_volume_gradient"), interpolate_volume_gradient, METH_VARARGS, NULL},
  {const_cast<char*>("interpolate_colormap"), interpolate_colormap, METH_VARARGS, NULL},
  {const_cast<char*>("set_outside_volume_colors"), set_outside_volume_colors, METH_VARARGS, NULL},

  /* mesh_edges.h */
  {const_cast<char*>("masked_edges"), (PyCFunction)masked_edges,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* spline.h */
  {const_cast<char*>("natural_cubic_spline"), (PyCFunction)natural_cubic_spline,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* squaremesh.h */
  {const_cast<char*>("principle_plane_edges"), (PyCFunction)principle_plane_edges,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* subdivide.h */
  {const_cast<char*>("subdivide_triangles"), (PyCFunction)subdivide_triangles,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("subdivide_mesh"), (PyCFunction)subdivide_mesh,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* pycontour.h */
  {const_cast<char*>("surface"), (PyCFunction)surface_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("scale_and_shift_vertices"), scale_and_shift_vertices,
   METH_VARARGS, NULL},
  {const_cast<char*>("scale_vertices"), scale_vertices, METH_VARARGS, NULL},
  {const_cast<char*>("shift_vertices"), shift_vertices, METH_VARARGS, NULL},
  {const_cast<char*>("affine_transform_vertices"), affine_transform_vertices,
   METH_VARARGS, NULL},
  {const_cast<char*>("reverse_triangle_vertex_order"),
   reverse_triangle_vertex_order, METH_VARARGS, NULL},

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

  /* tube.h */
  {const_cast<char*>("tube_geometry"), (PyCFunction)tube_geometry,
   METH_VARARGS|METH_KEYWORDS, NULL},

  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int image3d_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int image3d_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_image3d",
        NULL,
        sizeof(struct module_state),
        image3d_methods,
        NULL,
        image3d_traverse,
        image3d_clear,
        NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
extern "C" PyObject *
PyInit__image3d(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    
    if (module == NULL)
      return NULL;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_image3d.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}

}	// Image_3d
