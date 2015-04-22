# -----------------------------------------------------------------------------
# These routines are used by code outside the volume viewer module.
#

# -----------------------------------------------------------------------------
# Open volume data file, create a volume object and display it.
#
from .volume import open_volume_file, volume_list
from .volume import volume_list

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

# -----------------------------------------------------------------------------
# Mouse mode for moving planes.
#
from .moveplanes import planes_mouse_mode

# -----------------------------------------------------------------------------
# The active volume is the one highlighted in the volume dialog.  If the
# dialog has not been created (for example running Chimera with no graphical
# user interface) then there is no active volume.
#
#from volumedialog import active_volume, set_active_volume

# -----------------------------------------------------------------------------
# Display dialog to choose volume files and return volume objects to a
# callback function.
#
#from volumedialog import show_volume_file_browser

# -----------------------------------------------------------------------------
# Bounds of interactively drawn a outline box for selecting subregions.
#
#from volumedialog import subregion_selection_bounds

# -----------------------------------------------------------------------------
# Tk menu widget that contains an automatically updated list of open volume
# data sets.
#
#from volumemenu import Volume_Menu
