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

#include <iostream>			// use std::cerr for debugging
#include <Python.h>			// use PyObject

#include "capper.h"			// use compute_cap
#include "connected.h"			// use connected_triangles, ...
#include "convexity.h"			// use vertex_convexity
#include "measure.h"			// use enclosed_volume, surface_area, ...
#include "normals.h"			// use calculate_vertex_normals, invert_vertex_normals
#include "patches.h"			// use sharp_edge_patches
#include "refinemesh.h"			// use refine_mesh
#include "sasa.h"			// use surface_area_of_spheres
#include "smooth.h"			// use smooth_vertex_positions
#include "subdivide.h"			// use subdivide_triangles
#include "surfdist.h"			// use surface_distance
#include "triangulate.h"		// use triangulate_polygon
#include "tube.h"			// use tube_geometry

namespace Surface_Cpp
{

// ----------------------------------------------------------------------------
//
static struct PyMethodDef surface_cpp_methods[] =
{
  /* capper.h */
  {const_cast<char*>("compute_cap"),
   (PyCFunction)compute_cap,
   METH_VARARGS|METH_KEYWORDS,
R"(
compute_cap(plane_normal, plane_offset, varray, tarray)

Compute the portion of a plane inside a given surface.
Implemented in C++.

Returns
-------
cap_vertices : n by 3 array of float
cap_triangles : m by 3 array of int
)"
  },

// ----------------------------------------------------------------------------  
  /* connected.h */
  {const_cast<char*>("connected_triangles"),
   (PyCFunction)connected_triangles,
   METH_VARARGS|METH_KEYWORDS,
R"(
connected_triangles(triangles, tindex)

Return sorted array of triangle indices of triangles connected to the
specified triangle.  Two triangles are connected if they share a vertex.
The surface must be oriented and at most two triangles can share an edge.
The triangle array is triples of indices of vertices (m by 3, Numpy int32).
Implemented in C++.

Returns
-------
   triangles : 1d array of int
)"
  },

// ----------------------------------------------------------------------------
  {const_cast<char*>("triangle_vertices"),
   (PyCFunction)triangle_vertices,
   METH_VARARGS|METH_KEYWORDS,
R"(
triangle_vertices(triangles, tindices)

Return an array of vertex indices used the specified subset of triangles.

Returns
-------
vertex_indices : 1d array of int
)"
  },

// ----------------------------------------------------------------------------  
  {const_cast<char*>("connected_pieces"),
   (PyCFunction)connected_pieces,
   METH_VARARGS|METH_KEYWORDS,
R"(
connected_pieces(triangles)

Return each connected piece of a surface as a separate triangle array
and vertex array.  The return value is a tuple of pairs of vertex and
triangle index arrays.  Vertices connected by any sequence of triangle
edges are considered connected.
Implemented in C++.
)"
  },

// ----------------------------------------------------------------------------
  /* convexity.h */
  {const_cast<char*>("vertex_convexity"),
   (PyCFunction)vertex_convexity,
   METH_VARARGS|METH_KEYWORDS,
R"(
vertex_convexity(vertices, triangles, smoothing_iterations, convexity)

Compute convexity values for each vertex and save in the convexity array.
Convexity is defined as 2*pi - (vertex cone angle).
The surface should be closed so the triangles around each vertex define a cone.
Optional smoothing_iterations averages each vertex convexity value with neighbor
vertices connected by edges for the specified number of iterations.
The vertex array is triples of float values (n by 3, numpy float32).
The triangle array is triples of indices of vertices (m by 3, Numpy int32).
Implemented in C++.
)"
  },

// ----------------------------------------------------------------------------  
  /* measure.h */
  {const_cast<char*>("enclosed_volume"),
   (PyCFunction)enclosed_volume,
   METH_VARARGS|METH_KEYWORDS,
R"(
enclosed_volume(vertices, triangles)

If surface has hole then returned volume is computed by capping
boundary loops with fans centered at geometric center of loops.
Returns volume and hole count.
Implemented in C++.

Returns
-------
volume : float
hole_count : int
)"
  },

// ----------------------------------------------------------------------------
  {const_cast<char*>("surface_area"),
   (PyCFunction)surface_area,
   METH_VARARGS|METH_KEYWORDS,
R"(
surface_area(vertices, triangles)

Sum of triangle areas.
Implemented in C++.

Returns
-------
area : float
)"
  },

// ----------------------------------------------------------------------------
  {const_cast<char*>("vertex_areas"),
   (PyCFunction)vertex_areas,
   METH_VARARGS|METH_KEYWORDS,
R"(
vertex_areas(vertices, triangles, areas)

Accumulate 1/3 triangle area to each vertex.
Third parameter areas is a float array for returning vertex area values.
Implemented in C++.
)"
  },

// ----------------------------------------------------------------------------
  {const_cast<char*>("boundary_edges"),
   (PyCFunction)boundary_edges,
   METH_VARARGS|METH_KEYWORDS,
R"(
boundary_edges(triangles)

Returns N by 2 array of vertex indices for directed edges.
Implemented in C++.

Returns
-------
 vertex_pairs : N x 2 array of int
)"
  },

// ----------------------------------------------------------------------------
  {const_cast<char*>("boundary_loops"),
   (PyCFunction)boundary_loops,
   METH_VARARGS|METH_KEYWORDS,
R"(
boundary_loops(triangles)

Returns tuple of arrays of vertex indices, one array for each loop.
Implemented in C++.

Returns
-------
loops : tuple of 1d arrays of int
)"
  },

// ----------------------------------------------------------------------------
  {const_cast<char*>("boundary_edge_mask"),
   (PyCFunction)boundary_edge_mask,
   METH_VARARGS|METH_KEYWORDS,
R"(
boundary_edge_mask(triangles)

Returns mask values bits 0,1,2 are set if triangle edges 01, 12, 20 are on boundary.
Implemented in C++.

Returns
-------
edge_mask : 1d array of uint8, same length as triangle array
)"
  },

// ----------------------------------------------------------------------------  
  /* normals.h */
  {const_cast<char*>("calculate_vertex_normals"),
   (PyCFunction)calculate_vertex_normals,
   METH_VARARGS|METH_KEYWORDS,
R"(
calculate_vertex_normals(vertices, triangles)

Calculate vertex normals by adding normals for each triangle that uses
a vertex, and then normalizing the sum.
Implemented in C++.
)"
  },

// ----------------------------------------------------------------------------  
  {const_cast<char*>("invert_vertex_normals"),
   (PyCFunction)invert_vertex_normals,
   METH_VARARGS|METH_KEYWORDS,
R"(
invert_vertex_normals(normals, triangles)

Flip normals and reverse triangle vertex order.
Implemented in C++.
)"
  },

// ----------------------------------------------------------------------------  
  /* patches.h */
  {const_cast<char*>("sharp_edge_patches"),
   (PyCFunction)sharp_edge_patches,
   METH_VARARGS|METH_KEYWORDS,
R"(
sharp_edge_patches(vertices, normals, triangles, vertex_to_atom_index_map, atom_positions, atom_radii)

Split triangles to create sharp boundaries equidistant between atoms.
Equidistant means an equal number of atom radii away.
Implemented in C++.

Returns
-------
vertices : n by 3 array of float
normals : n by 3 array of float
triangles : m by 3 array of int
vertex_to_atom_index_map : 1d array of int
)"
  },

// ----------------------------------------------------------------------------  
  {const_cast<char*>("unique_vertex_map"),
   (PyCFunction)unique_vertex_map,
   METH_VARARGS|METH_KEYWORDS,
R"(
unique_vertex_map(vertices)

Map vertex indices to unique vertex indices so vertices at the same point
are treated as one.  This is used for connected piece calculations.
Implemented in C++.

Returns
-------
vertex_map : 1d array of int, maps vertex index to unique vertex index.
)"
  },

// ----------------------------------------------------------------------------    
  /* refinemesh.h */
  {const_cast<char*>("refine_mesh"),
   (PyCFunction)refine_mesh,
   METH_VARARGS|METH_KEYWORDS,
R"(
refine_mesh(vertices, triangles, subdivision_factor)

Modify a planar surface triangulation to create uniform size triangles
suited for vertex coloring.
Implemented in C++.

Returns
-------
ref_vertices : n by 3 array of float
ref_triangles : n by 3 array of int
)"
  },

// ----------------------------------------------------------------------------    
  /* sasa.h */
  {const_cast<char*>("surface_area_of_spheres"),
   (PyCFunction)surface_area_of_spheres,
   METH_VARARGS|METH_KEYWORDS,
R"(
surface_area_of_spheres(centers, radii, areas)

Compute surface area of union of solid sphere.
Third argument areas contains areas contributed by each sphere
Can fail in degenerate cases giving area -1 for spheres with failed area calculation.
Implemented in C++.
)"
  },

// ----------------------------------------------------------------------------    
  {const_cast<char*>("estimate_surface_area_of_spheres"),
   (PyCFunction)estimate_surface_area_of_spheres,
   METH_VARARGS|METH_KEYWORDS,
R"(
estimate_surface_area_of_spheres(centers, radii, sphere_points, point_weights, areas)

Use points on sphere, count how many are inside other spheres
to estimate surface area of union of solid spheres.
Third argument areas contains areas contributed by each sphere
Implemented in C++.
)"
  },

// ----------------------------------------------------------------------------    
  /* smooth.h */
  {const_cast<char*>("smooth_vertex_positions"),
   (PyCFunction)smooth_vertex_positions,
   METH_VARARGS|METH_KEYWORDS,
R"(
smooth_vertex_positions(vertices, triangles, smoothing_factor, smoothing_iterations)

Move surface vertices towards the average of neighboring vertices
to give the surface a smoother appearance.  Modifies vertices numpy array.
Implemented in C++.
)"
  },

// ----------------------------------------------------------------------------    
  /* subdivide.h */
  {const_cast<char*>("subdivide_triangles"),
   (PyCFunction)subdivide_triangles,
   METH_VARARGS|METH_KEYWORDS,
R"(
subdivide_triangles(vertices, triangles, normals)

Divide each triangle into 4 triangles placing new vertices at edge midpoints.
Implemented in C++.

Returns
-------
vertices : n by 3 array of float
triangles : m by 3 array of int
normals : n by 3 array of float
)"

// ----------------------------------------------------------------------------    
  },
  {const_cast<char*>("subdivide_mesh"),
   (PyCFunction)subdivide_mesh,
   METH_VARARGS|METH_KEYWORDS,
R"(
subdivide_mesh(vertices, triangles, normals, edge_length)

Divide triangle into smaller triangles so that edges are shorter
than the specified the maximum edge length.
Implemented in C++.

Returns
-------
vertices : n by 3 array of float
triangles : m by 3 array of int
normals : n by 3 array of float
)"
  },

// ----------------------------------------------------------------------------    
  /* surfdist.h */
  {const_cast<char*>("surface_distance"),
   (PyCFunction)surface_distance,
   METH_VARARGS|METH_KEYWORDS,
R"(
surface_distance(points, vertices, triangles, distances = None)

Compute the closest distance from a point to a surface.  Do this for
each point in a list.  The distance, closest point, and side of the closest
triangle that the given point lies on is returned in an N by 5 float32 array.
Side +1 is the right-handed normal clockwise vertex traversal, while -1
indicates the opposite side.  This is for determining if the given point
is inside or outside the surface.  If a distance array (N by 5) is passed
as an argument, it will only be modified by distances less those.  If no
distance array is provided, a newly allocated one will be returned.
Implemented in C++.

Returns
-------
distances : n by 5 array of float
)"
  },

// ----------------------------------------------------------------------------    
  /* triangulate.h */
  {const_cast<char*>("triangulate_polygon"),
   (PyCFunction)triangulate_polygon,
   METH_VARARGS|METH_KEYWORDS,
R"(
triangulate_polygon(loops, normal, vertices)

Triangulate a set of loops in a plane."
Implemented in C++.

Returns
-------
triangles : m by 3 array of int
)"
  },

// ----------------------------------------------------------------------------    
  /* tube.h */
  {const_cast<char*>("tube_geometry"),
   (PyCFunction)tube_geometry,
   METH_VARARGS|METH_KEYWORDS,
R"(
tube_geometry(path, tangents, cross_section, cross_section_normals)

Calculates tube surface geometry from a center-line path.
Arguments path and tangents are n by 3 float arrays,
and the cross section arguments are m by 3 arrays.
Implemented in C++.

Returns
-------
vertices : n by 3 array of float
normals : n by 3 array of float
triangles : m by 3 array of int
)"
  },

// ----------------------------------------------------------------------------    
  {const_cast<char*>("tube_geometry_colors"),
   (PyCFunction)tube_geometry_colors,
   METH_VARARGS|METH_KEYWORDS,
R"(
tube_geometry_colors(colors, segment_subdivisions, circle_subdivisions, start_divisions, end_divisions)

Computes vertex colors for a tube with geometry determined by tube_geometry()
Each segment can have a separate color. Colors argument is N by 4 array.
A segment is is the interval between segment_subdivisions+1 path points not
including points at ends of the path specified by start/end divisions.
Arguments other than colors are integers.
Implemented in C++.

Returns
-------
colors : N by 4 array of uint8
)"
  },

// ----------------------------------------------------------------------------    
  {const_cast<char*>("tube_triangle_mask"),
   (PyCFunction)tube_triangle_mask,
   METH_VARARGS|METH_KEYWORDS,
R"(
tube_triangle_mask(segment_mask, segment_subdivisions, circle_subdivisions, start_divisions, end_divisions)

Computes triangle mask to show only specified segments of a tube generated with tube_geometry().
Segments are defined in the same way as for the tube_geometry_colors() routine.
The input segment mask is a uint8 length N array, and output is a uint8 numpy array
with length equal to number of triangles generated by tube_geometry().
Implemented in C++.

Returns
-------
triangle_mask : 1d array of uint8
)"
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
PyMODINIT_FUNC
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
