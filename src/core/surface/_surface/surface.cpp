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
   METH_VARARGS|METH_KEYWORDS,
   "Return sorted array of triangle indices of triangles connected to the\n"
   "specified triangle.  Two triangles are connected if they share a vertex.\n"
   "The surface must be oriented and at most two triangles can share an edge.\n"
   "The triangle array is triples of indices of vertices (m by 3, Numpy int32).\n"
   "\n"
   "Args: triangle_array, int tindex\n"
  },
  {const_cast<char*>("triangle_vertices"), (PyCFunction)triangle_vertices,
   METH_VARARGS|METH_KEYWORDS,
   "Return an array of vertex indices used the specified subset of triangles.\n"
   "\n"
   "Args: triangle_array, triangle_list (int array)\n"
  },
  {const_cast<char*>("connected_pieces"), (PyCFunction)connected_pieces,
   METH_VARARGS|METH_KEYWORDS,
   "Return each connected piece of a surface as a separate triangle array\n"
   "and vertex array.  The return value is a tuple of pairs of vertex and\n"
   "triangle index arrays.  Vertices connected by any sequence of triangle\n"
   "edges are considered connected.\n"
   "\n"
   "Args: triangle_array (N by 3 int)\n"
  },

  /* measure.h */
  {const_cast<char*>("enclosed_volume"), (PyCFunction)enclosed_volume,
   METH_VARARGS|METH_KEYWORDS,
   "If surface has hole then returned volume is computed by capping\n"
   "boundary loops with fans centered at geometric center of loops.\n"
   "Returns volume and hole count.\n"
   "\n"
   "Args: vertices, triangles\n"
  },
  {const_cast<char*>("surface_area"), (PyCFunction)surface_area,
   METH_VARARGS|METH_KEYWORDS,
   "Sum of triangle areas.\n"
   "\n"
   "Args: vertices, triangles\n"
   "\n"
  },
  {const_cast<char*>("vertex_areas"), (PyCFunction)vertex_areas,
   METH_VARARGS|METH_KEYWORDS,
   "Accumulate 1/3 triangle area to each vertex.\n"
   "\n"
   "Args: vertices, triangles, areas\n"
  },
  {const_cast<char*>("boundary_edges"), (PyCFunction)boundary_edges,
   METH_VARARGS|METH_KEYWORDS,
   "Returns N by 2 array of vertex indices for directed edges.\n"
   "\n"
   "Args: triangles\n"
  },
  {const_cast<char*>("boundary_loops"), (PyCFunction)boundary_loops,
   METH_VARARGS|METH_KEYWORDS,
   "Returns tuple of arrays of vertex indices, one array for each loop.\n"
   "\n"
   "Args: triangles\n"
  },

  /* normals.h */
  {const_cast<char*>("calculate_vertex_normals"), (PyCFunction)calculate_vertex_normals,
   METH_VARARGS|METH_KEYWORDS,
   "Args: vertex_array, triangle_array"
  },
  {const_cast<char*>("invert_vertex_normals"), (PyCFunction)invert_vertex_normals,
   METH_VARARGS|METH_KEYWORDS,
   "Args: normals_array, triangle_array"
  },

  /* parse_stl.h */
  {const_cast<char*>("parse_stl"), (PyCFunction)parse_stl,
   METH_VARARGS|METH_KEYWORDS,
   "Read a binary STL file passed in as a byte array and return\n"
   "4 values: the header comment, unique vertices, corresponding normals\n"
   "and triangles (triples of vertex indices).\n"
   "\n"
   "Args: binary STL as byte array\n"
   },

  /* patches.h */
  {const_cast<char*>("sharp_edge_patches"), (PyCFunction)sharp_edge_patches,
   METH_VARARGS|METH_KEYWORDS,
   "Split triangles to create sharp boundaries equidistant between atoms.\n"
   "\n"
   "Args: vertices, normals, triangles, vertex to atom index array, atom positions.\n"
   "\n"
   "Returns subdivide vertices, normals, triangles, vertex to atom.\n"
  },
  {const_cast<char*>("unique_vertex_map"), (PyCFunction)unique_vertex_map,
   METH_VARARGS|METH_KEYWORDS,
   "Map vertex indices to unique vertex indices so vertices at the same point\n"
   "are treated as one.  This is used for connected piece calculations.\n"
   "\n"
   "Args: vertex array, n by 3 float\n"
   "\n"
   "Returns numpy int array, length n, mapping vertex index to unique vertex index.\n"
  },

  /* sasa.h */
  {const_cast<char*>("surface_area_of_spheres"), (PyCFunction)surface_area_of_spheres,
   METH_VARARGS|METH_KEYWORDS,
   "Compute surface area of union of solid sphere.\n"
   "Can fail in degenerate cases returning -1 for spheres with failed area calculation.\n"
   "\n"
   "Args: centers, radii, areas.\n"
   "\n"
   "Returns: array of areas contributed by each sphere\n"
  },
  {const_cast<char*>("estimate_surface_area_of_spheres"), (PyCFunction)estimate_surface_area_of_spheres,
   METH_VARARGS|METH_KEYWORDS,
   "Use points on sphere, count how many are inside other spheres\n"
   "to estimate surface area of union of solid spheres.\n"
   "\n"
   "Args: centers, radii, sphere_points, point_weights, areas\n"
   "\n"
   "Returns: array of areas contributed by each sphere.\n"
  },

  /* subdivide.h */
  {const_cast<char*>("subdivide_triangles"), (PyCFunction)subdivide_triangles,
   METH_VARARGS|METH_KEYWORDS,
   "Divide each triangle into 4 triangles placing new vertices at edge midpoints.\n"
   "\n"
   "Args: vertices, triangles, normals\n"
   "\n"
   "Returns: vertices triangles, normals\n"
  },
  {const_cast<char*>("subdivide_mesh"), (PyCFunction)subdivide_mesh,
   METH_VARARGS|METH_KEYWORDS,
   "Divide triangle into smaller triangles so that edges are shorter\n"
   "than the specified the maximum edge length.\n"
   "\n"
   "Args: vertices, triangles, normals, edge_length\n"
   "\n"
   "Returns: vertices triangles, normals\n"
  },

  /* tube.h */
  {const_cast<char*>("tube_geometry"), (PyCFunction)tube_geometry,
   METH_VARARGS|METH_KEYWORDS,
   "Calculates tube surface geometry from a center-line path.\n"
   "\n"
   "Args: path, tangents, cross_section, cross_section_normals -- all n by 3 float arrays\n"
   "\n"
   "Returns: vertices, normals, triangles\n"
},
  {const_cast<char*>("tube_geometry_colors"), (PyCFunction)tube_geometry_colors,
   METH_VARARGS|METH_KEYWORDS,
   "Args: colors, segment_subdivisions, circle_subdivisions, start_divisions, end_divisions\n"
   "\n"
   "Returns: N by 4 numpy array of RGBA colors.\n"
  },
  {const_cast<char*>("tube_triangle_mask"), (PyCFunction)tube_triangle_mask,
   METH_VARARGS|METH_KEYWORDS,
   "Computes triangle mask to show only specified segments of a tube generated with tube_geometry().\n"
   "\n"
   "Args: segment_mask (uint8 length N array), segment_subdivisions, circle_subdivisions, start_divisions, end_divisions\n"
   "\n"
   "Returns: uint8 numpy array equal to number of triangles generated by tube_geometry().\n"
  },

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
