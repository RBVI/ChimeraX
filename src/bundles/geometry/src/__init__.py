# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Load libarrays shared library needed by _geometry and
# other C++ modules to parse numpy arrays.  This code loads
# the library into the ChimeraX process so it does not need
# to be on the runtime loader search path.
import chimerax.arrays

from .place import Place, Places, identity
from .place import rotation, quaternion_rotation, vector_rotation, translation, scale, z_align
from .place import product, orthonormal_frame, interpolate_rotation, transform_planes, look_at
from .place import multiply_transforms
from .vector import interpolate_points, inner_product, normalize_vector, normalize_vectors, ray_segment
from .vector import planes_as_4_vectors, distance, cross_product, cross_products, norm, distance_squared
from .vector import angle, dihedral, dihedral_point, length
from .vector import clip_segment
from .matrix import linear_combination, project_to_axis
from .bounds import sphere_bounds, union_bounds, Bounds, point_bounds
from .bounds import copies_bounding_box, copy_tree_bounds, clip_bounds
from ._geometry import natural_cubic_spline
from ._geometry import sphere_axes_bounds, spheres_in_bounds, bounds_overlap
from ._geometry import find_close_points, find_closest_points, find_close_points_sets
from ._geometry import closest_sphere_intercept, closest_cylinder_intercept, closest_triangle_intercept
from ._geometry import segment_intercepts_spheres, points_within_planes
from ._geometry import cylinder_rotations, half_cylinder_rotations, cylinder_rotations_x3d
from ._geometry import distances_from_origin, distances_parallel_to_axis, distances_perpendicular_to_axis
from ._geometry import fill_small_ring, fill_6ring
from .align import align_points
from .symmetry import cyclic_symmetry_matrices
from .symmetry import dihedral_symmetry_matrices
from .symmetry import tetrahedral_symmetry_matrices, tetrahedral_orientations
from .symmetry import octahedral_symmetry_matrices
from .symmetry import helical_symmetry_matrices, helical_symmetry_matrix
from .symmetry import translation_symmetry_matrices
from .symmetry import recenter_symmetries
from .icosahedron import icosahedral_symmetry_matrices, icosahedron_angles
from .icosahedron import coordinate_system_names as icosahedral_orientations
from .icosahedron import coordinate_system_transform as icosahedral_coordinate_system_transform
from .spline import arc_lengths
from .adaptive_tree import AdaptiveTree
from .plane import Plane, PlaneNoIntersectionError

from chimerax.core.toolshed import BundleAPI
class _GeometryBundleAPI(BundleAPI):
    @staticmethod
    def get_class(class_name):
        if class_name == 'Place':
            return Place
        elif class_name == 'Places':
            return Places
        elif class_name == 'Plane':
            return Plane

bundle_api = _GeometryBundleAPI()
