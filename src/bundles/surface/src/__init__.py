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

from .sasa import spheres_surface_area
from .split import split_surfaces
from .shapes import sphere_geometry, sphere_geometry2, cylinder_geometry, dashed_cylinder_geometry, cone_geometry, box_geometry
from .area import surface_area, enclosed_volume, surface_volume_and_area
from .gridsurf import ses_surface_geometry

# Make sure _surface can runtime link shared library libarrays.
import chimerax.arrays

from ._surface import subdivide_triangles, vertex_areas
from ._surface import surface_area_of_spheres, estimate_surface_area_of_spheres
from ._surface import calculate_vertex_normals, invert_vertex_normals
from ._surface import connected_triangles, triangle_vertices
from ._surface import sharp_edge_patches, unique_vertex_map, connected_pieces
from ._surface import boundary_edges, compute_cap, triangulate_polygon, refine_mesh, boundary_loops
from ._surface import boundary_edge_mask
from ._surface import vertex_convexity
from ._surface import smooth_vertex_positions

from .dust import largest_blobs_triangle_mask
from .gaussian import gaussian_surface
from .cap import update_clip_caps, remove_clip_caps
from .topology import check_surface_topology
from .colorgeom import color_radial, color_cylindrical, color_height
from .colorvol import color_sample, color_electrostatic, color_gradient, color_surfaces_by_map_value
from .combine import combine_geometry, combine_geometry_vnt, combine_geometry_vntc
from .combine import combine_geometry_xvnt, combine_geometry_vtp
from .combine import combine_geometry_xvntctp, combine_geometry_vte
from .surfacecmds import surface, surface_show_patches, surface_hide_patches
from .sop import surface_zone

from chimerax.core.toolshed import BundleAPI

class _SurfaceBundle(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        from . import settings
        settings.settings = settings._SurfaceSettings(session, "surfaces")

        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready',
                lambda *args, ses=session: settings.register_settings_options(ses))

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
        elif (command_name.startswith('surface') or
              command_name.startswith('sop') or
              command_name.startswith('~surface')):
            from . import surfacecmds
            surfacecmds.register_command(logger)
            from . import check
            check.register_command(logger)
            
    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'Hide Dust':
            from . import dustgui
            ti = dustgui.show_hide_dust_panel(session)
        elif tool_name == 'Surface Zone':
            from . import zonegui
            ti = zonegui.show_surface_zone_panel(session)
        elif tool_name == 'Color Zone':
            from . import colorzonegui
            ti = colorzonegui.show_color_zone_panel(session)
        elif tool_name == 'Measure Volume and Area':
            from . import areagui
            ti = areagui.show_volume_area_panel(session)
        elif tool_name == 'Surface Color':
            from . import surfcolorgui
            ti = surfcolorgui.show_surface_color_panel(session)
        else:
            ti = None
        return ti

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        from .cap import ClipCap
        from .colorgeom import CylinderColor, HeightColor, RadialColor
        from .colorvol import GradientColor, VolumeColor
        from .dust import Redust
        from .zone import ZoneMask
        from .colorzone import ZoneColor
        from .updaters import SurfaceUpdaters
        ct = {
            'ClipCap': ClipCap,
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

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_command import OpenerInfo
        class ColladaOpenerInfo(OpenerInfo):
            def open(self, session, data, file_name, **kw):
                from . import collada
                return collada.read_collada_surfaces(session, data, file_name)
        return ColladaOpenerInfo()

bundle_api = _SurfaceBundle()
