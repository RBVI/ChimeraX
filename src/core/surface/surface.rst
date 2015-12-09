..  vim: set expandtab shiftwidth=4 softtabstop=4:

surface: Triangulated surface calculations
==========================================

Routines for calculating surface triangulations and properties of triangulated surfaces.

Surfaces are represented using numpy arrays of vertices (N by 3 array, xyz coodinates, float32),
and a numpy array of triangles which are triples of indices into the vertex list
(M by 3 array, vertex indices, int32).  For surface lighting, normal vectors at each vertex are used
(N by 3 array, unit vectors, float32).  The vertex, triangle and normal arrays are
sometimes called varray, tarray, narray, and sometimes vertices, triangles, normals.

.. automodule:: chimerax.core.surface.sasa
    :members:

.. automodule:: chimerax.core.surface.shapes
    :members:

.. automodule:: chimerax.core.surface.area
    :members:

.. automodule:: chimerax.core.surface.gridsurf
    :members:

.. automodule:: chimerax.core.surface._surface
    :members: connected_triangles, triangle_vertices, connected_pieces, enclosed_volume, surface_area, vertex_areas, boundary_edges, boundary_loops, calculate_vertex_normals, invert_vertex_normals, parse_stl, sharp_edge_patches, unique_vertex_map, surface_area_of_spheres, estimate_surface_area_of_spheres, subdivide_triangles, subdivide_mesh, tube_geometry, tube_geometry_colors, tube_triangle_mask

.. automodule:: chimerax.core.surface.dust
    :members:

.. automodule:: chimerax.core.surface.gaussian
    :members:
