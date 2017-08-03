# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .sasa import spheres_surface_area
from .split import split_surfaces
from .shapes import sphere_geometry, cylinder_geometry, dashed_cylinder_geometry, cone_geometry, box_geometry
from .area import surface_area, enclosed_volume, surface_volume_and_area
from .gridsurf import ses_surface_geometry
from ._surface import subdivide_triangles, vertex_areas
from ._surface import surface_area_of_spheres, estimate_surface_area_of_spheres
from ._surface import calculate_vertex_normals, invert_vertex_normals
from ._surface import connected_triangles, sharp_edge_patches, unique_vertex_map
from ._surface import compute_cap
from .dust import largest_blobs_triangle_mask
from .gaussian import gaussian_surface
from .cap import update_clip_caps
from .topology import check_surface_topology
