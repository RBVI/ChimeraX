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
# Wrap image data as grid data for displaying surface, meshes, and volumes.
#
from chimerax.map.data import GridData

# -----------------------------------------------------------------------------
#
def dicom_grids(paths, log = None, verbose = False):
  from .dicom_format import find_dicom_series, DicomData
  series = find_dicom_series(paths, log = log, verbose = verbose)
  grids = []
  derived = []	# For grouping derived series with original series
  sgrids = {}
  for s in series:
    if not s.has_image_data:
      if s.attributes.get('SOPClassUID').name == 'RT Structure Set Storage':
        from .dicom_contours import DicomContours
        DicomContours(log.session, s.paths[0])
      continue
    d = DicomData(s)
    if d.mode == 'RGB':
      # Create 3-channels for RGB series
      cgrids = [ ]
      colors = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
      suffixes = [' red', ' green', ' blue']
      for channel in (0,1,2):
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
      if s.attributes.get('BitsAllocated') == 1:
        g.binary = True		# Use initial thresholds for binary segmentation
      rs = getattr(s, 'refers_to_series', None)
      if rs:
        # If this associated with another series (e.g. is a segmentation), make
        # it a channel together with that associated series.
        derived.append((g, rs))
      else:
        sgrids[s] = gg = [g]
        grids.append(gg)

  # Group derived series with the original series
  channel_colors = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
  for g,rs in derived:
    sg = sgrids[rs]
    if len(sg) == 1:
      sg[0].channel = 1
    sg.append(g)
    g.channel = len(sg)
    g.rgba = channel_colors[(g.channel-2) % len(channel_colors)]

  # Show only first group of grids
  for gg in grids[1:]:
    for g in gg:
      g.show_on_open = False
      
  return grids

# -----------------------------------------------------------------------------
#
class DicomGrid(GridData):

  def __init__(self, d, time = None, channel = None):

    self.dicom_data = d

    GridData.__init__(self, d.data_size, d.value_type,
                      d.data_origin, d.data_step, rotation = d.data_rotation,
                      path = d.paths, name = d.name,
                      file_type = 'dicom', time = time, channel = channel)

    self.multichannel = (channel is not None)

    self.initial_plane_display = True
    self.initial_thresholds_linear = True
    self.ignore_pad_value = d.pad_value

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from chimerax.map.data.readarray import allocate_array
    m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    c = self.channel if self.multichannel else None
    self.dicom_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                                self.time, c, m, progress)
    return m
