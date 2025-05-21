..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

surface: Triangulated surface calculations
==========================================

Routines for calculating surface triangulations and properties of triangulated surfaces.

Surfaces are represented using numpy arrays of vertices (N by 3 array, xyz coodinates, float32),
and a numpy array of triangles which are triples of indices into the vertex list
(M by 3 array, vertex indices, int32).  For surface lighting, normal vectors at each vertex are used
(N by 3 array, unit vectors, float32).  The vertex, triangle and normal arrays are
sometimes called varray, tarray, narray, and sometimes vertices, triangles, normals.

.. automodule:: chimerax.surface.sasa
    :members:
    :show-inheritance:

.. automodule:: chimerax.surface.shapes
    :members:
    :show-inheritance:

.. automodule:: chimerax.surface.area
    :members:
    :show-inheritance:

.. automodule:: chimerax.surface.gridsurf
    :members:
    :show-inheritance:

.. automodule:: chimerax.surface._surface
    :members: connected_triangles, triangle_vertices, connected_pieces, enclosed_volume, surface_area, vertex_areas, boundary_edges, boundary_loops, calculate_vertex_normals, invert_vertex_normals, sharp_edge_patches, unique_vertex_map, surface_area_of_spheres, estimate_surface_area_of_spheres, subdivide_triangles, subdivide_mesh, tube_geometry, tube_geometry_colors, tube_triangle_mask, compute_cap, triangulate_polygon, refine_mesh, boundary_loops, boundary_edge_mask, vertex_convexity, smooth_vertex_positions
    :noindex: boundary_loops

.. automodule:: chimerax.surface.dust
    :members:
    :show-inheritance:

.. automodule:: chimerax.surface.gaussian
    :members:
    :show-inheritance:
