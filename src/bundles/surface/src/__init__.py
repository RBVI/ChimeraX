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
from .shapes import sphere_geometry, sphere_geometry2, cylinder_geometry, dashed_cylinder_geometry, cone_geometry, box_geometry
from .area import surface_area, enclosed_volume, surface_volume_and_area
from .gridsurf import ses_surface_geometry
from ._surface import subdivide_triangles, vertex_areas
from ._surface import surface_area_of_spheres, estimate_surface_area_of_spheres
from ._surface import calculate_vertex_normals, invert_vertex_normals
from ._surface import connected_triangles, sharp_edge_patches, unique_vertex_map, connected_pieces
from ._surface import boundary_edges, compute_cap, triangulate_polygon
from ._surface import vertex_convexity
from ._surface import smooth_vertex_positions
from .dust import largest_blobs_triangle_mask
from .gaussian import gaussian_surface
from .cap import update_clip_caps, remove_clip_caps
from .topology import check_surface_topology
from .colorgeom import color_radial, color_cylindrical, color_height
from .colorvol import color_sample, color_electrostatic, color_gradient, color_surfaces_by_map_value
from .surfacecmds import surface, surface_hide
from .sop import surface_zone

from chimerax.core.toolshed import BundleAPI

class _SurfaceBundle(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        if command_name.startswith('color'):
            from . import colorcmds
            colorcmds.register_color_subcommand(command_name, logger)
        elif command_name.startswith('measure'):
            from . import area
            area.register_measure_subcommand(command_name, logger)
        elif command_name == 'volume splitbyzone':
            from . import colorzone
            colorzone.register_volume_split_command(logger)
        else:
            from . import surfacecmds
            surfacecmds.register_command(logger)

    @staticmethod
    def open_file(session, stream, file_name):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        from . import collada
        return collada.read_collada_surfaces(session, stream, file_name)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        from .colorgeom import CylinderColor, HeightColor, RadialColor
        from .colorvol import GradientColor, VolumeColor
        from .dust import Redust
        from .zone import ZoneMask
        from .colorzone import ZoneColor
        from .updaters import SurfaceUpdaters
        ct = {
            'CylinderColor': CylinderColor,
            'GradientColor': GradientColor,
            'HeightColor': HeightColor,
            'RadialColor': RadialColor,
            'Redust': Redust,
            'SurfaceUpdaters': SurfaceUpdaters,
            'VolumeColor': VolumeColor,
            'ZoneColor': ZoneColor,
            'ZoneMask': ZoneMask,
        }
        return ct.get(class_name)

bundle_api = _SurfaceBundle()
