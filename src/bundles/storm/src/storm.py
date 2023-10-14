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

"""
storm: Read STORM microscopy point lists
========================================
"""
def read_storm(session, filename, name):
    """Create density maps from STORM microscopy point list files.

    :param filename: either the name of a file or a file-like object

    Extra arguments are ignored.
    """

    if hasattr(filename, 'read'):
        # it's really a file-like object
        f = filename
    else:
        f = open(filename, 'rb')

    f.readline()	# Column headers

    # Old data:
    # Columns: Channel	Xc	Yc	Zc	Height	Area	Width	Phi	Ax	BG
    
    # New data:
    # Columns channel	Xc	Yc	Zc	h	Width	BG
    # h = Gaussian height, width = FWHM, BG = background.
    points = {}
    while True:
        line = f.readline()
        if not line:
            break
        fields = line.split()
        channel = int(fields[0])
        xyzhaw = [float(v) for v in fields[1:7]]
        points.setdefault(channel, []).append(xyzhaw)

    maps = []
    step = 10		# TODO: Should adjust step based on data extent
    pad = 10*step
    cutoff_range = 5	# Standard deviations
    grid = None
    from chimerax.map.molmap import bounding_grid, add_gaussians
    from chimerax.map import volume_from_grid_data
    for channel in sorted(points.keys()):
        plist = points[channel]
        from numpy import array, float32, sqrt
        xyzhwb = array(plist, float32)
        xyz = xyzhwb[:,:3].copy()
        if grid is None:
            grid = bounding_grid(xyz, step, pad)
        else:
            grid = matching_grid(grid)
        grid.name = 'channel %d' % channel
        if channel == 1:
            grid.rgba = (0, 1.0, 0, 1.0)
        elif channel == 2:
            grid.rgba = (1.0, 0, 1.0, 1.0)
        weights = xyzhwb[:,3]
        from math import sqrt, log
        fwhm_sdev = 2 * sqrt(2*log(2))	# Convert full width half max to standard deviations
        sdev = xyzhwb[:,4] / fwhm_sdev
        sdev /= 20	# TODO: Seems the width column is not FWHM.
        add_gaussians(grid, xyz, weights, sdev, cutoff_range, normalize=False)
        v = volume_from_grid_data(grid, session, style = 'image')
        v.show()
        maps.append(v)
        
    return maps, ("Opened STORM file %s containing %d channels, %d points"
                  % (f.name, len(points), sum((len(plist) for plist in points.values()),0)))

def matching_grid(grid):
    from numpy import zeros, float32
    matrix = zeros(grid.full_matrix().shape, float32)
    from chimerax.map_data import ArrayGridData
    g = ArrayGridData(matrix, grid.origin, grid.step)
    return g
