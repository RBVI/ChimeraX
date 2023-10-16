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

# -----------------------------------------------------------------------------
# Python readers for array file formats.
#

from .griddata import GridData, GridSubregion
from .arraygrid import ArrayGridData
from .subsample import SubsampledGrid
from .fileformats import file_formats, MapFileFormat, electrostatics_types, FileFormatError
from .fileformats import open_file, FileFormatError, UnknownFileType, save_grid_data
from .progress import ProgressReporter
from .readarray import allocate_array

# -----------------------------------------------------------------------------
# Routines to find tri-linear interpolated values of a 3d array.
#
from .arrays import interpolate_volume_data, interpolate_volume_gradient

from .arrays import MatrixValueStatistics, invert_matrix
from .arrays import grid_indices, zone_masked_grid_data, zone_mask
from .arrays import zone_mask, masked_grid_data
from .arrays import surface_level_enclosing_volume

from .regions import points_ijk_bounds, bounding_box, clamp_region, box_corners
