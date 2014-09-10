# -----------------------------------------------------------------------------
# Performs a Fourier transform on current Volume Viewer data set producing
# a new data set that is shown in Volume Viewer.
#
def fourier_transform(v = None, step = None, subregion = None, model_id = None,
                      phase = False):

  if v is None:
    from VolumeViewer import active_volume
    v = active_volume()
    if v is None:
      return

  m = v.matrix(step = step, subregion = subregion)
  from numpy.fft import fftn
  cftm = fftn(m)      # Complex128 result, same array size as input

  from numpy import absolute, angle, float32
  if phase:
    aftm = angle(cftm).astype(float32)          # Radians
  else:
    aftm = absolute(cftm).astype(float32)
    aftm *= 1.0/aftm.size       # Normalization
    aftm[0,0,0] = 0     # Constant term often huge making histogram hard to use
  cftm = None           # Release memory
  ftm = fftshift(aftm)
  aftm = None           # Release memory

  # Place FT centered on data, scaled to keep same volume
  xyz_min, xyz_max = v.xyz_bounds()
  xyz_center = map(lambda a,b: 0.5*(a+b), xyz_min, xyz_max)
  ijk_size = list(ftm.shape)
  ijk_size.reverse()

  if step is None:
    ijk_step = v.region[2]
  elif isinstance(step, int):
    ijk_step = (step,step,step)
  else:
    ijk_step = step
  xyz_size = map(lambda a,b: a-b, xyz_max, xyz_min)
  vol = xyz_size[0]*xyz_size[1]*xyz_size[2]
  cell_size = map(lambda a,b: a*b, v.data.step, ijk_step)
  cell_vol = cell_size[0]*cell_size[1]*cell_size[2]
  scale = pow(vol*cell_vol, 1.0/3)
  step = map(lambda a: scale/a, xyz_size)
  origin = map(lambda c,s,z: c-0.5*s*z, xyz_center, step, ijk_size)

  from VolumeData import Array_Grid_Data
  ftd = Array_Grid_Data(ftm, origin, step)
  ftd.name = v.name + (' FT phase' if phase else ' FT')
  from VolumeViewer import volume_from_grid_data
  ftr = volume_from_grid_data(ftd, show_data = False, model_id = model_id)
  ftr.copy_settings_from(v, copy_thresholds = False,  copy_colors = False,
                         copy_region = False)
  ftr.initialize_thresholds()
  ftr.set_parameters(show_outline_box = True)
  ftr.show()
  
  v.unshow()          # Hide original map

  return ftr

# -----------------------------------------------------------------------------
# Recenter to put 0 component at center of map.
#
# There is a numpy routine with the same name that is incredibly memory
# inefficient.
#
def fftshift(m):

  k,j,i = m.shape
  from numpy import empty
  sm = empty(m.shape, m.dtype)
  for ko1, ko2 in ((0,k/2), (k/2,k)):
    for jo1, jo2 in ((0,j/2), (j/2,j)):
      for io1, io2 in ((0,i/2), (i/2,i)):
        sm[k-ko2:k-ko1,j-jo2:j-jo1,i-io2:i-io1] = m[ko1:ko2,jo1:jo2,io1:io2]
  return sm
