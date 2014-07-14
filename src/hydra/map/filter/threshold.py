# -----------------------------------------------------------------------------
# Clamp map values.
#
def threshold(volume, minimum = None, set_minimum = None,
              maximum = None, set_maximum = None,
              step = 1, subregion = None, modelId = None):

  tg = threshold_grid(volume, minimum, set_minimum, maximum, set_maximum,
                      step, subregion)
  from VolumeViewer import volume_from_grid_data
  tv = volume_from_grid_data(tg, show_data = False, model_id = modelId)
  tv.copy_settings_from(volume)
  tv.show()
  
  volume.unshow()          # Hide original map
  
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

  from VolumeData import Array_Grid_Data
  d = v.data
  if v.name.endswith('thresholded'): name = v.name
  else:                         name = '%s thresholded' % v.name
  tg = Array_Grid_Data(m, origin, step, d.cell_angles, d.rotation,
                       name = name)
  return tg
