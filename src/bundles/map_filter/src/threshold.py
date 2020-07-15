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
# Clamp map values.
#
def threshold(volume, minimum = None, set_minimum = None,
              maximum = None, set_maximum = None,
              step = 1, subregion = None, modelId = None, session = None):

  tg = threshold_grid(volume, minimum, set_minimum, maximum, set_maximum,
                      step, subregion)
  from chimerax.map import volume_from_grid_data
  tv = volume_from_grid_data(tg, model_id = modelId, session = session)
  tv.copy_settings_from(volume)
  
  volume.display = False          # Hide original map
  
  return tv

# -----------------------------------------------------------------------------
#
def threshold_grid(volume, minimum = None, set_minimum = None,
                   maximum = None, set_maximum = None,
                   step = 1, subregion = None, region = None):

  v = volume
  if region is None:
    region = v.subregion(step, subregion)

  origin, step = v.region_origin_and_step(region)

  m = v.region_matrix(region).copy()

  import numpy
  from numpy import array, putmask
  if not minimum is None:
    t = array(minimum, m.dtype)
    if set_minimum is None or set_minimum == minimum:
      numpy.maximum(m, t, m)
    else:
      putmask(m, m < t, array(set_minimum, m.dtype))
  if not maximum is None:
    t = array(maximum, m.dtype)
    if set_maximum is None or set_maximum == maximum:
      numpy.minimum(m, t, m)
    else:
      putmask(m, m > t, array(set_maximum, m.dtype))

  from chimerax.map_data import ArrayGridData
  d = v.data
  if v.name.endswith('thresholded'): name = v.name
  else:                         name = '%s thresholded' % v.name
  tg = ArrayGridData(m, origin, step, d.cell_angles, d.rotation,
                     name = name)
  return tg
