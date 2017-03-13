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

from .place import Place, Places, identity, rotation, vector_rotation, translation, scale
from .place import product, orthonormal_frame, interpolate_rotation, transform_planes, look_at
from .vector import interpolate_points, inner_product, normalize_vector, ray_segment
from .vector import planes_as_4_vectors, distance, cross_product, norm, distance_squared
from .vector import angle, dihedral, dihedral_point
from .matrix import linear_combination
from .bounds import sphere_bounds, union_bounds, Bounds, point_bounds
from .bounds import copies_bounding_box, copy_tree_bounds, clip_bounds
from ._geometry import natural_cubic_spline
from ._geometry import sphere_axes_bounds, spheres_in_bounds, bounds_overlap
from ._geometry import find_close_points, find_closest_points, find_close_points_sets
from ._geometry import closest_sphere_intercept, closest_cylinder_intercept, closest_triangle_intercept
from ._geometry import segment_intercepts_spheres, points_within_planes
from ._geometry import cylinder_rotations, cylinder_rotations_x3d
from .align import align_points
from .symmetry import cyclic_symmetry_matrices
from .symmetry import dihedral_symmetry_matrices
from .symmetry import tetrahedral_symmetry_matrices, tetrahedral_orientations
from .symmetry import octahedral_symmetry_matrices
from .symmetry import helical_symmetry_matrices, helical_symmetry_matrix
from .symmetry import translation_symmetry_matrices
from .symmetry import recenter_symmetries
from .icosahedron import icosahedral_symmetry_matrices
from .icosahedron import coordinate_system_names as icosahedral_orientations
from .spline import arc_lengths
