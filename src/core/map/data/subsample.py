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
# Subsampled_Grid wraps a set of Grid_Data objects.  One of the wrapped grid
# objects is the primary data and the others are subsamples.
#

from . import Grid_Data

# -----------------------------------------------------------------------------
#
class Subsampled_Grid(Grid_Data):

  def __init__(self, primary_grid_data):

    pg = primary_grid_data

    self.available_subsamplings = {(1,1,1): pg}

    settings = pg.settings()
    Grid_Data.__init__(self, path = pg.path, file_type = pg.file_type,
                       grid_id = pg.grid_id, **settings)

  # ---------------------------------------------------------------------------
  #
  def _get_data_cache(self):
    return self.__dict__['data_cache']
  def _set_data_cache(self, dc):
    self.__dict__['data_cache'] = dc
    for g in self.available_subsamplings.values():
      g.data_cache = dc
  data_cache = property(_get_data_cache, _set_data_cache)

  # ---------------------------------------------------------------------------
  # It is the caller's responsibility to verify that the passed in subsampled
  # data has a matching set of components and a valid subsampling size.
  #
  def add_subsamples(self, grid_data, cell_size):

    csize = tuple(cell_size)
    if csize in self.available_subsamplings:
      del self.available_subsamplings[csize]
    self.available_subsamplings[csize] = grid_data

  # ---------------------------------------------------------------------------
  # Return the subsampling and size of subsampled matrix for the requested
  # ijk_step.
  #
  def choose_subsampling(self, ijk_step):

    compatible = []
    for step, grid in self.available_subsamplings.items():
      if (ijk_step[0] % step[0] == 0 and
          ijk_step[1] % step[1] == 0 and
          ijk_step[2] % step[2] == 0):
        e = ((ijk_step[0] // step[0]) *
             (ijk_step[1] // step[1]) *
             (ijk_step[2] // step[2]))
        compatible.append((e, step, grid.size))

    if len(compatible) == 0:
      return (1,1,1), self.size

    subsampling, size = min(compatible)[1:]
    return subsampling, size
    
  # ---------------------------------------------------------------------------
  #
  def matrix(self, ijk_origin = (0,0,0), ijk_size = None,
             ijk_step = (1,1,1), progress = None, from_cache_only = False,
             subsampling = None):

    if subsampling is None:
      subsampling, ss_size = self.choose_subsampling(ijk_step)
      
    if ijk_size == None:
      ijk_size = self.size

    d = self.available_subsamplings[tuple(subsampling)]
    origin = [i//s for i,s in zip(ijk_origin, subsampling)]
    size = [(i+s-1)//s for i,s in zip(ijk_size, subsampling)]
    step = [i//s for i,s in zip(ijk_step, subsampling)]
    m = d.matrix(origin, size, step, progress, from_cache_only)
    return m

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):

    m = self.matrix(ijk_origin, ijk_size, ijk_step, progress)
    return m
        
  # ---------------------------------------------------------------------------
  #
  def cached_data(self, ijk_origin, ijk_size, ijk_step):

    m = self.matrix(ijk_origin, ijk_size, ijk_step, from_cache_only = True)
    return m

  # ---------------------------------------------------------------------------
  #
  def clear_cache(self):

    for d in self.available_subsamplings.values():
      d.clear_cache()
