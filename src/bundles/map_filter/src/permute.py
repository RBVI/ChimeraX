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
# Permute volume axes.
#
def permute_axes(v, axis_order = (0,1,2),
                 step = None, subregion = None, model_id = None):

  d = v.grid_data(subregion, step, mask_zone = False)
  pd = PermutedGrid(d, axis_order)
  from chimerax.map import volume_from_grid_data
  pv = volume_from_grid_data(pd, v.session, model_id = model_id)
  return pv

# -----------------------------------------------------------------------------
#
from chimerax.map_data import GridData
class PermutedGrid(GridData):
  
  def __init__(self, grid_data, axis_order):

    self.grid_data = g = grid_data
    self.axis_order = ao = axis_order
    settings = g.settings(size = permute(g.size,ao),
                          origin = permute(g.origin,ao),
                          step = permute(g.step,ao),
                          name = g.name + ' permuted')
    GridData.__init__(self, **settings)
    
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    ao = self.axis_order
    iao = inverse_permutation(ao)
    porigin = permute(ijk_origin, iao)
    psize = permute(ijk_size, iao)
    pstep = permute(ijk_step, iao)
    data = self.grid_data.read_matrix(porigin, psize, pstep, progress)
    # Array axis order is z,y,x so reverse axes.
    rao = list(ao)
    rao.reverse()
    rao = [2-a for a in rao]
    dt = data.transpose(rao)
    return dt

# -----------------------------------------------------------------------------
#
def inverse_permutation(p):

  ip = [None] * len(p)
  for a,pa in enumerate(p):
    ip[pa] = a
  return tuple(ip)

# -----------------------------------------------------------------------------
#
def permute(v,p):

  return tuple([v[pa] for pa in p])
