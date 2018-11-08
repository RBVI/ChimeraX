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
# Wrap DOCK grid data for displaying surface, meshes, and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class DockGrid(GridData):

  def __init__(self, dock_data, component_name):

    d = dock_data
    self.dock_data = d
    self.component_name = component_name

    path = d.path
    from os.path import basename
    name = basename(path) + ' ' + component_name

    GridData.__init__(self, d.data_size, d.value_type(component_name),
                      d.data_origin, d.data_step,
                      name = name, path = path, file_type = 'dock',
                      grid_id = component_name,
                      default_color = d.color(component_name))
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    m = self.dock_data.read_matrix(self.component_name,
                                   ijk_origin, ijk_size, ijk_step, progress)
    return m
    
# -----------------------------------------------------------------------------
#
def read_dock_file(path):

  from . import dock_format
  dock_data = dock_format.Dock_Data(path)

  grids = []
  for cname in dock_data.component_names:
    g = DockGrid(dock_data, cname)
    grids.append(g)

  return grids
