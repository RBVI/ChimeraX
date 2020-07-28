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
# Scale, shift, and change value type of volume values.
#
def scaled_volume(v, scale = 1, sd = None, rms = None, shift = 0, type = None,
                 step = None, subregion = None, model_id = None, session = None):

  if not sd is None or not rms is None:
    m = v.full_matrix()
    from chimerax.map.volume import mean_sd_rms
    mean, sdev, rmsv = mean_sd_rms(m)
    if not rms is None and rmsv > 0:
      scale = (1.0 if scale is None else scale) * rms / rmsv
    if not sd is None and sdev > 0:
      shift = -mean if shift is None else (-mean + shift)
      scale = (1.0 if scale is None else scale) * sd / sdev
    
  sg = scaled_grid(v, scale, shift, type, subregion, step)
  from chimerax.map import volume_from_grid_data
  sv = volume_from_grid_data(sg, session, model_id = model_id)
  sv.copy_settings_from(v, copy_thresholds = False)

  v.display = False          # Hide original map

  return sv

# -----------------------------------------------------------------------------
#
def scaled_grid(v, scale, shift, type, subregion = None, step = 1,
                region = None):
  if region is None:
    d = v.grid_data(subregion, step, mask_zone = False)
  else:
    from chimerax.map_data import GridSubregion
    d = GridSubregion(v.data, *region)
  sd = ScaledGrid(d, scale, shift, type)
  return sd

# -----------------------------------------------------------------------------
#
from chimerax.map_data import GridData
class ScaledGrid(GridData):
  
  def __init__(self, grid_data, scale, shift, value_type):

    self.grid_data = g = grid_data
    self.scale = scale
    self.shift = shift
    self.value_type = vt = (value_type or g.value_type)
    settings = g.settings(value_type=vt, name = g.name + ' scaled')
    GridData.__init__(self, **settings)
    
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    data = self.grid_data.matrix(ijk_origin, ijk_size, ijk_step, progress)
    s = self.scale
    o = self.shift
    t = self.value_type
    if s == 1 and o == 0:
      dt = data
    else:
      # Convert to float32, then scale and shift.
      from numpy import float32, multiply, add
      dt = data.astype(float32)
      if o != 0:
        add(dt, o, dt)
      if s != 1:
        multiply(dt, s, dt)

    if t == dt.dtype:
      d = dt
    else:
      from numpy import dtype
      if dtype(t).kind in 'iu':
        # Clamp integer types to limit values.
        from numpy import empty, iinfo
        d = empty(dt.shape, t)
        di = iinfo(t)
        dmin, dmax = di.min, di.max
        if dt.dtype.kind in 'iu':
          # Clip limits get cast to dt type by dt.clip().
          # int16 -> uint16 incorrectly gave all 65535 values when clipped.
          dti = iinfo(dt.dtype)
          dmin = max(dti.min, dmin)
          dmax = min(dti.max, dmax)
        dt.clip(dmin, dmax, d)
      else:
        d = dt.astype(t)

    return d
