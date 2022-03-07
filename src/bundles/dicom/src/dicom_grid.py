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

from chimerax.map_data.readarray import allocate_array
from chimerax.map_data import GridData
from .dicom_format import DicomData
from .dicom_format import find_dicom_series

def dicom_grids(paths, log = None, verbose = False):
    series = find_dicom_series(paths, log = log, verbose = verbose)
    grids = dicom_grids_from_series(series)
    return grids

def dicom_grids_from_series(series):
    grids = []
    derived = []	# For grouping derived series with original series
    sgrids = {}
    for s in series:
        if not s.has_image_data:
            continue
        d = DicomData(s)
        if d.mode == 'RGB':
            # Create 3-channels for RGB series
            cgrids = []
            colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
            suffixes = [' red', ' green', ' blue']
            for channel in (0, 1, 2):
                g = DicomGrid(d, channel=channel)
                g.name += suffixes[channel]
                g.rgba = colors[channel]
                cgrids.append(g)
            grids.append(cgrids)
        elif s.num_times > 1:
            # Create time series for series containing multiple times as frames
            tgrids = []
            for t in range(s.num_times):
                g = DicomGrid(d, time=t)
                g.series_index = t
                tgrids.append(g)
            grids.append(tgrids)
        else:
            # Create single channel, single time series.
            g = DicomGrid(d)
            rs = getattr(s, 'refers_to_series', None)
            if rs:
                # If this associated with another series (e.g. is a segmentation), make
                # it a channel together with that associated series.
                derived.append((g, rs))
            else:
                sgrids[s] = gg = [g]
                grids.append(gg)
    # Group derived series with the original series
    channel_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    for g, rs in derived:
        sg = sgrids[rs]
        if len(sg) == 1:
            sg[0].channel = 1
        sg.append(g)
        g.channel = len(sg)
        g.rgba = channel_colors[(g.channel - 2) % len(channel_colors)]
        if not g.dicom_data.origin_specified:
            # Segmentation may not have specified an origin
            g.set_origin(sg[0].origin)
    # Show only first group of grids
    for gg in grids[1:]:
        for g in gg:
            g.show_on_open = False
    return grids

class DicomGrid(GridData):
    initial_rendering_options = {'projection_mode': '3d',
                                 'colormap_on_gpu': True,
                                 'full_region_on_gpu': True}

    def __init__(self, d, time = None, channel = None):
        self.dicom_data = d
        GridData.__init__(self, d.data_size, d.value_type,
                          d.data_origin, d.data_step, rotation = d.data_rotation,
                          path = d.paths, name = d.name,
                          file_type = 'dicom', time = time, channel = channel)
        if d.files_are_3d:
            # For fast access of multiple planes this reading method avoids
            # opening same dicom file many times to read planes.  50x faster in tests.
            self.read_matrix = self.dicom_read_matrix
        else:
            # For better caching if we read the whole plane, this method caches the
            # whole plane even if only part of the plane is needed.
            self.read_xy_plane = self.dicom_read_xy_plane
        self.multichannel = (channel is not None)
        self.initial_plane_display = True
        s = d.dicom_series
        if s.attributes.get('BitsAllocated') == 1 or s.dicom_class == 'Segmentation Storage':
            self.binary = True		# Use initial thresholds for binary segmentation
            self.initial_image_thresholds = [(0.5, 0), (1.5, 1)]
        else:
            self.initial_image_thresholds = [(-1000, 0.0), (300, 0.9), (3000, 1.0)]
        self.ignore_pad_value = d.pad_value

    # ---------------------------------------------------------------------------
    # If GridData.read_xy_plane() uses this method then whole planes are cached
    # even when a partial plane is requested.  The whole DICOM planes are always
    # read.  Caching them helps performance when say an xz-plane is being read.
    #
    def dicom_read_xy_plane(self, k):
        c = self.channel if self.multichannel else None
        m = self.dicom_data.read_plane(k, self.time, c)
        return m

    # ---------------------------------------------------------------------------
    # If GridData.read_matrix() uses this method it only caches the actual requested
    # data.  For multiframe DICOM files this is much faster than reading separate
    # planes where the dicom data has to be opened once for each plane.
    # In a 512 x 512 x 235 test with 3d dicom segmentations this is 50x faster
    # than reading xy planes one at a time.
    #
    def dicom_read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
        c = self.channel if self.multichannel else None
        self.dicom_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                                    self.time, c, m, progress)
        return m
