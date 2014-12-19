# -----------------------------------------------------------------------------
# Permute volume axes.
#
def permute_axes(v = None, axis_order = (0,1,2),
                 step = None, subregion = None, model_id = None):

  if v is None:
    from VolumeViewer import active_volume
    v = active_volume()
    if v is None:
      return

  d = v.grid_data(subregion, step, mask_zone = False)
  pd = Permuted_Grid(d, axis_order)
  import VolumeViewer
  pv = VolumeViewer.volume_from_grid_data(pd, model_id = model_id)
  return pv

# -----------------------------------------------------------------------------
#
from ..data import Grid_Data
class Permuted_Grid(Grid_Data):
  
  def __init__(self, grid_data, axis_order):

    self.grid_data = g = grid_data
    self.axis_order = ao = axis_order
    Grid_Data.__init__(self, permute(g.size,ao), g.value_type,
                       permute(g.origin,ao), permute(g.step,ao),
                       g.cell_angles, g.rotation, g.symmetries,
                       name = g.name + ' permuted', default_color = g.rgba)
    
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
