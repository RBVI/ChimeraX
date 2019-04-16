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

# -----------------------------------------------------------------------------
# Create and show a volume madel from a GridData object as defined by the
# data module.
#
from .volume import volume_from_grid_data

# -----------------------------------------------------------------------------
# A grid data object combined with graphical display state is a Volume object.
# Many methods of Volume objects (interpolation, coloring, thresholds, ...)
# are available.
#
from .volume import Volume

# -----------------------------------------------------------------------------
# Map contouring and distance maps.
#
from ._map import contour_surface, sphere_surface_distance
from ._map import interpolate_colormap, set_outside_volume_colors
from ._map import extend_crystal_map
from ._map import moments, affine_scale
from ._map import local_correlation
from ._map import linear_combination
from ._map import covariance_sum
from ._map import offset_range, box_cuts

# -----------------------------------------------------------------------------
# Control whether maps are pickable with mouse.
#
from .volume import maps_pickable

# -----------------------------------------------------------------------------
# Mouse modes for moving planes and changing contour level
#
from .moveplanes import PlanesMouseMode
from .mouselevel import ContourLevelMouseMode

# -----------------------------------------------------------------------------
# Routines to register map file formats, database fetch, and volume command.
#
from .volume import register_map_file_formats, add_map_format
from .volume import MapChannelsModel, MultiChannelSeries
from .eds_fetch import register_eds_fetch
from .emdb_fetch import register_emdb_fetch
from .volumecommand import register_volume_command
from .molmap import register_molmap_command
from .mapargs import MapArg, MapsArg, Float1or3Arg

# -----------------------------------------------------------------------------
#
from chimerax.core.toolshed import BundleAPI

class _MapBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        # 'start_tool' is called to start an instance of the tool
        from . import volume_viewer
        return volume_viewer.show_volume_dialog(session)

    @staticmethod
    def open_file(session, stream, file_name):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        return None


    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from chimerax import map
        if command_name == 'volume' or command_name == 'vop':
            map.register_volume_command(logger)
        elif command_name == 'molmap':
            map.register_molmap_command(logger)
        elif command_name == 'measure mapstats':
            from . import measure
            measure.register_measure_mapstats_command(logger)

    @staticmethod
    def initialize(session, bundle_info):
        """Register file formats, commands, and database fetch."""
        from chimerax import map
        map.register_map_file_formats(session)
        map.register_eds_fetch()
        map.register_emdb_fetch()
        if session.ui.is_gui:
            from . import mouselevel, moveplanes, windowing
            mouselevel.register_mousemode(session)
            moveplanes.register_mousemode(session)
            windowing.register_mousemode(session)


    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        from . import Volume, MapChannelsModel, MultiChannelSeries
        from .volume import VolumeSurface
        from .session import GridDataState
        ct = {
            'GridDataState': GridDataState,
            'MapChannelsModel': MapChannelsModel,
            'MultiChannelSeries': MultiChannelSeries,
            'Volume': Volume,
            'VolumeSurface': VolumeSurface,
        }
        return ct.get(class_name)

bundle_api = _MapBundle()
