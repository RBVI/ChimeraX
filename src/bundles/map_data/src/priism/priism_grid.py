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
# Wrap Priism microscope image data as grid data for displaying
# surface, meshes, and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class PriismGrid(GridData):

  def __init__(self, priism_data, wd, time):

    self.wavelength_data = wd
    self._time = time

    from os.path import basename
    if wd.wavelength == 0:
      name = basename(priism_data.path)
    else:
      name = '%s %d' % (basename(priism_data.path), wd.wavelength)

    size = priism_data.data_size
    xyz_step = priism_data.data_step
    xyz_origin = [a * b for a,b in zip(priism_data.data_origin, xyz_step)]
    value_type = wd.element_type

    wavelength = wd.wavelength
    opacity = 1
    wcolors = {460: (0, .7, .7, opacity),          # cyan
               535: (0, .7, 0, opacity),           # green
               605: (.7, 0, 0, opacity),           # red
               690: (0, 0, .7, opacity),           # blue
               }
    if wavelength in wcolors:
      initial_color = wcolors[wavelength]
    else:
      initial_color = (.7, .7, .7, opacity)         # white

    GridData.__init__(self, size, value_type,
                      xyz_origin, xyz_step,
                      name = name, path = priism_data.path,
                      file_type = 'priism', grid_id = str(wd.wave_index),
                      default_color = initial_color)

    self.num_times = priism_data.num_times
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    return self.wavelength_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                                            self._time, progress)
  
# -----------------------------------------------------------------------------
#
def read_priism_file(path):

  from . import priism_format
  priism_data = priism_format.Priism_Data(path)

  grids = []
  nt = priism_data.num_times
  for t in range(nt):
    for c,wd in enumerate(priism_data.wavelength_data):
      g = PriismGrid(priism_data, wd, t)
      if priism_data.num_waves > 1:
        g.channel = c
      if nt > 1:
        g.time = t
        g.series_index = t
      grids.append(g)

  return grids
