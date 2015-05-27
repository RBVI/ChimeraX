# -----------------------------------------------------------------------------
# Add voxels in 2x2x2, 3x3x3 or IxJxK bins and divide by number of voxels
# in bin for average value.  Useful for reducing noise.
#
def bin(v, bin_size = (2,2,2),
        step = None, subregion = None, model_id = None, session = None):

  bd = bin_grid(v, bin_size, step, subregion)
  from .. import volume_from_grid_data
  bv = volume_from_grid_data(bd, session, model_id = model_id, show_data = False)

  bv.copy_settings_from(v, copy_region = False)
  bv.show()
  v.unshow()          # Hide original map

  return bv

# -----------------------------------------------------------------------------
#
def bin_grid(v, bin_size = (2,2,2), step = 1, subregion = None, region = None):

  d = v.grid_data(subregion, step, mask_zone = False)
  bin_size = [min(a,b) for a,b in zip(bin_size, d.size)]
  g = Binned_Grid(d, bin_size)
  return g

# -----------------------------------------------------------------------------
#
from ..data import Grid_Data
class Binned_Grid(Grid_Data):
  
  def __init__(self, grid_data, bin_size):

    self.grid_data = g = grid_data
    self.bin_size = bin_size
    size = [s//b for s,b in zip(g.size, bin_size)]
    step = [s*b for s,b in zip(g.step, bin_size)]
    origin = [o+0.5*(s-gs) for o,s,gs in zip(g.origin, step, g.step)]
    Grid_Data.__init__(self, size, g.value_type, origin, step,
                       g.cell_angles, g.rotation, g.symmetries,
                       name = g.name + ' binned', default_color = g.rgba)
    
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    b = self.bin_size
    bo = [o*s for o,s in zip(ijk_origin, b)]
    bs = [z*s for z,s in zip(ijk_size, b)]
    d = self.grid_data.read_matrix(bo, bs, (1,1,1), progress)

    bi, bj, bk = b
    from numpy import zeros, single as floatc
    bdf = zeros(d[::bk,::bj,::bi].shape, floatc)
    for k in range(bk):
      for j in range(bj):
        for i in range(bi):
          bdf += d[k::bk,j::bj,i::bi]
    bdf /= (bi*bj*bk)
    si,sj,sk = ijk_step
    bd = bdf[::sk,::sj,::si].astype(d.dtype)
    return bd
