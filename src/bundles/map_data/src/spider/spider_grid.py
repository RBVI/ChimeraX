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
# Wrap SPIDER volume data as grid data for displaying surface, meshes, and
# volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class SPIDERGrid(GridData):

  def __init__(self, path):

    from . import spider_format
    d = spider_format.SPIDER_Data(path)
    self.spider_data = d

    origin = tuple(a * b for a,b in zip(d.data_origin, d.data_step))

    GridData.__init__(self, d.data_size, origin = origin, step = d.data_step,
                      path = path, file_type = 'spider')

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    return self.spider_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                                        progress)
