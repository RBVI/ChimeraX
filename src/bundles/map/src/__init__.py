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
from .volume import VolumeSurface, VolumeImage

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
from ._map import high_indices
from ._map import indices_to_colors

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
from .volume import add_map_format
from .volume import MapChannelsModel, MultiChannelSeries
from .volumecommand import register_volume_command
from .molmap import register_molmap_command
from .mapargs import MapArg, MapsArg, Float1or3Arg, ValueTypeArg, MapRegionArg, MapStepArg

# -----------------------------------------------------------------------------
#
from chimerax.core.toolshed import BundleAPI

class _MapBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        # 'start_tool' is called to start an instance of the tool
        if tool_name == 'Volume Viewer':
            from . import volume_viewer
            return volume_viewer.show_volume_dialog(session)
        elif tool_name == 'Map Coordinates':
            from .coords_gui import show_coords_panel
            show_coords_panel(session)

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
        elif command_name == 'segmentation':
            map.register_segmentation_command(logger)

    @staticmethod
    def initialize(session, bundle_info):
        """Register file formats, commands, and database fetch."""
        if session.ui.is_gui:
            from . import mouselevel, moveplanes, windowing, tiltedslab
            mouselevel.register_mousemode(session)
            moveplanes.register_mousemode(session)
            windowing.register_mousemode(session)
            tiltedslab.register_mousemode(session)


    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        from . import Volume, MapChannelsModel, MultiChannelSeries
        from .volume import VolumeSurface, VolumeImage
        from .session import GridDataState
        ct = {
            'GridDataState': GridDataState,
            'MapChannelsModel': MapChannelsModel,
            'MultiChannelSeries': MultiChannelSeries,
            'Volume': Volume,
            'VolumeSurface': VolumeSurface,
            'VolumeImage': VolumeImage,
        }
        return ct.get(class_name)

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            fetches = ['eds', 'edsdiff', 'emdb']
            if name in fetches:
                from . import eds_fetch, emdb_fetch
                fetcher = {
                    'eds': eds_fetch.fetch_eds_map,
                    'edsdiff': eds_fetch.fetch_edsdiff_map,
                    'emdb': emdb_fetch.fetch_emdb
                }[name]
                from chimerax.open_command import FetcherInfo
                class Info(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache,
                            fetcher=fetcher, **kw):
                        return fetcher(session, ident, ignore_cache=ignore_cache, **kw)
            else:
                from chimerax.open_command import OpenerInfo
                class Info(OpenerInfo):
                    def open(self, session, path, file_name,
                            _name=session.data_formats[name].nicknames[0], **kw):
                        from .volume import open_map
                        return open_map(session, path, format=_name, **kw)

                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg, IntArg, StringArg
                        return {
                            'array_name': StringArg,
                            'channel': IntArg,
                            'verbose': BoolArg,
                            'vseries': BoolArg,
                        }
        else:
            from chimerax.save_command import SaverInfo
            class Info(SaverInfo):
                def save(self, session, path, _name=name, **kw):
                    from .volume import save_map
                    save_map(session, path, _name, **kw)

                @property
                def save_args(self, _name=name):
                    from .mapargs import MapRegionArg, Int1or3Arg
                    from chimerax.core.commands import BoolArg, ModelsArg, EnumOf, \
                        RepeatOf, IntArg, ListOf
                    args = { 'models': ModelsArg }
                    if _name == "Chimera map":
                        args.update({
                            'append': BoolArg,
                            'base_index': IntArg,
                            'chunk_shapes': ListOf(EnumOf(
                                ('zyx','zxy','yxz','yzx','xzy','xyz'))),
                            'compress': BoolArg,
                            'compress_method': EnumOf(('zlib', 'lzo', 'bzip2', 'blosc',
                                'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc',
                                'blosc:snappy', 'blosc:zlib', 'blosc:zstd')),
                            'compress_shuffle': BoolArg,
                            'compress_level': IntArg,
                            'mask_zone': BoolArg,
                            'region': MapRegionArg,
                            'subsamples': RepeatOf(Int1or3Arg),
                            'step': Int1or3Arg,
                        })
                    return args

                def save_args_widget(self, session):
                    from chimerax.save_command import SaveModelOptionWidget
                    return SaveModelOptionWidget(session, 'Map', Volume)

                def save_args_string_from_widget(self, widget):
                    return widget.options_string()

        return Info()

bundle_api = _MapBundle()
