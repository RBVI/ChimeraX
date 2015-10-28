# vim: set expandtab shiftwidth=4 softtabstop=4:

# -----------------------------------------------------------------------------
# Create and show a volume madel from a Grid_Data object as defined by the
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

# -----------------------------------------------------------------------------
# Mouse modes for moving planes and changing contour level
#
from .moveplanes import PlanesMouseMode
from .mouselevel import ContourLevelMouseMode

# -----------------------------------------------------------------------------
# Routines to register map file formats, database fetch, and volume command.
#
from .volume import register_map_file_readers
from .eds_fetch import register_eds_fetch
from .emdb_fetch import register_emdb_fetch
from .volumecommand import register_volume_command
from .molmap import register_molmap_command
from .mapargs import MapArg
