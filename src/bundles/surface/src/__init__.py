# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Surface algorithms are not needed merely to initialize the bundle. Resolve
# public exports on first use, retaining their original functions and classes.
import chimerax.arrays  # _surface needs libarrays to be linkable before loading.

_surface_exports = {
    'spheres_surface_area': 'sasa',
    'split_surfaces': 'split',
    'sphere_geometry': 'shapes',
    'sphere_geometry2': 'shapes',
    'cylinder_geometry': 'shapes',
    'dashed_cylinder_geometry': 'shapes',
    'cone_geometry': 'shapes',
    'box_geometry': 'shapes',
    'surface_area': 'area',
    'enclosed_volume': 'area',
    'surface_volume_and_area': 'area',
    'ses_surface_geometry': 'gridsurf',
    'subdivide_triangles': '_surface',
    'vertex_areas': '_surface',
    'surface_area_of_spheres': '_surface',
    'estimate_surface_area_of_spheres': '_surface',
    'calculate_vertex_normals': '_surface',
    'invert_vertex_normals': '_surface',
    'connected_triangles': '_surface',
    'triangle_vertices': '_surface',
    'sharp_edge_patches': '_surface',
    'unique_vertex_map': '_surface',
    'connected_pieces': '_surface',
    'boundary_edges': '_surface',
    'compute_cap': '_surface',
    'triangulate_polygon': '_surface',
    'refine_mesh': '_surface',
    'boundary_loops': '_surface',
    'boundary_edge_mask': '_surface',
    'vertex_convexity': '_surface',
    'smooth_vertex_positions': '_surface',
    'surface_distance': '_surface',
    'largest_blobs_triangle_mask': 'dust',
    'gaussian_surface': 'gaussian',
    'update_clip_caps': 'cap',
    'remove_clip_caps': 'cap',
    'check_surface_topology': 'topology',
    'color_radial': 'colorgeom',
    'color_cylindrical': 'colorgeom',
    'color_height': 'colorgeom',
    'color_sample': 'colorvol',
    'color_electrostatic': 'colorvol',
    'color_gradient': 'colorvol',
    'color_surfaces_by_map_value': 'colorvol',
    'combine_geometry': 'combine',
    'combine_geometry_vnt': 'combine',
    'combine_geometry_vntc': 'combine',
    'combine_geometry_xvnt': 'combine',
    'combine_geometry_vtp': 'combine',
    'combine_geometry_xvntctp': 'combine',
    'combine_geometry_vtc': 'combine',
    'combine_geometry_vte': 'combine',
    'surface': 'surfacecmds',
    'surface_show_patches': 'surfacecmds',
    'surface_hide_patches': 'surfacecmds',
    'surface_zone': 'sop',
}

_surface_modules = (
    'sasa', 'split', 'shapes', 'area', 'gridsurf', 'dust', 'gaussian',
    'cap', 'topology', 'colorgeom', 'colorvol', 'combine', 'surfacecmds', 'sop',
)


def __getattr__(name):
    from importlib import import_module
    if name in _surface_exports:
        module = import_module('.' + _surface_exports[name], __name__)
        value = getattr(module, name)
    elif name in _surface_modules:
        value = import_module('.' + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_surface_exports) | set(_surface_modules))


__all__ = [*_surface_exports, *_surface_modules, 'BundleAPI', 'bundle_api', 'chimerax']

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
            from . import measure_contactarea
            measure_contactarea.register_contactarea_command(logger)
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
            from . import hidefarblobs
            hidefarblobs.register_command(logger)
            
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
