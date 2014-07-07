#include <iostream>			// use std::cerr for debugging
#include <Python.h>			// use PyObject

#include "arrayops.h"			// use value_ranges
#include "parsecif.h"			// use parse_mmcif_file
#include "parsepdb.h"			// use parse_pdb_file
#include "pdb_bonds.h"			// use molecule_bonds
#include "sasa.h"			// use surface_area_of_spheres
#include "spline.h"			// use natural_cubic_spline
#include "tube.h"			// use tube_geometry

namespace Molecule_Cpp
{

// ----------------------------------------------------------------------------
//
static struct PyMethodDef molecule_cpp_methods[] =
{
  /* arrayops.h */
  {const_cast<char*>("value_ranges"), (PyCFunction)value_ranges,
   METH_VARARGS|METH_KEYWORDS},
  {const_cast<char*>("contiguous_intervals"), (PyCFunction)contiguous_intervals,
   METH_VARARGS|METH_KEYWORDS},
  {const_cast<char*>("mask_intervals"), (PyCFunction)mask_intervals,
   METH_VARARGS|METH_KEYWORDS},
  {const_cast<char*>("duplicate_midpoints"), (PyCFunction)duplicate_midpoints,
   METH_VARARGS|METH_KEYWORDS},

  /* parsecif.h */
  {const_cast<char*>("parse_mmcif_file"), (PyCFunction)parse_mmcif_file,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* parsepdb.h */
  {const_cast<char*>("parse_pdb_file"), (PyCFunction)parse_pdb_file,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("element_radii"), (PyCFunction)element_radii,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("atom_sort_order"), (PyCFunction)atom_sort_order,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("residue_ids"), (PyCFunction)residue_ids,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* pdb_bonds.h */
  {const_cast<char*>("molecule_bonds"), (PyCFunction)molecule_bonds,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("initialize_bond_templates"), (PyCFunction)initialize_bond_templates,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* sasa.h */
  {const_cast<char*>("surface_area_of_spheres"), (PyCFunction)surface_area_of_spheres,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("estimate_surface_area_of_spheres"), (PyCFunction)estimate_surface_area_of_spheres,
   METH_VARARGS|METH_KEYWORDS, NULL},

  /* spline.h */
  {const_cast<char*>("natural_cubic_spline"), (PyCFunction)natural_cubic_spline,
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

static int molecule_cpp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int molecule_cpp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "molecule_cpp",
        NULL,
        sizeof(struct module_state),
        molecule_cpp_methods,
        NULL,
        molecule_cpp_traverse,
        molecule_cpp_clear,
        NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
extern "C" PyObject *
PyInit_molecule_cpp(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    
    if (module == NULL)
      return NULL;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("molecule_cpp.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}

}	// Molecule_Cpp namespace
