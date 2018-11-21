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
# Wrap DeltaVision image data as grid data for displaying surface, meshes, and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class DeltaVisionGrid(GridData):

  def __init__(self, data, channel, time):
    d = data
    self.dv_data = d
    GridData.__init__(self, d.data_size, d.element_type,
                      d.data_origin, d.data_step, d.cell_angles,
                      channel=channel, time=time, path = d.path)
    self.series_index = time
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
    return self.dv_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                                    channel=self.channel, time=self.time,
                                    progress=progress)


# -----------------------------------------------------------------------------
#
def read_deltavision_grids(path):
    from .deltavision_format import DeltaVision_Data
    d = DeltaVision_Data(path)
    grids = [DeltaVisionGrid(d, c, t) for c in range(d.nchannels) for t in range(d.ntimes)]

    # Set default channel colors for multichannel data.
    if d.nchannels > 1:
        default_colors = ((1,0,0,1),(0,1,0,1),(0,0,1,0),(0,1,1,1),(1,1,0,1),(1,0,1,1))
        for g in grids:
            g.rgba = default_colors[g.channel % len(default_colors)]

    return grids
