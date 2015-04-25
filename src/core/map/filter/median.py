# -----------------------------------------------------------------------------
# 3x3x3 median filter.
#
def median_filter(volume, bin_size = 3, iterations = 1,
                  step = 1, subregion = None, modelId = None):

  mg = median_grid(volume, bin_size, iterations, step, subregion)
  from VolumeViewer import volume_from_grid_data
  mv = volume_from_grid_data(mg, show_data = False, model_id = modelId)
  mv.copy_settings_from(volume, copy_region = False)
  mv.show()
  
  volume.unshow()          # Hide original map
  
  return mv

# -----------------------------------------------------------------------------
#
def median_grid(volume, bin_size = 3, iterations = 1,
                step = 1, subregion = None, region = None):

  v = volume
  if region is None:
    region = v.subregion(step, subregion)

  origin, step = v.region_origin_and_step(region)

  vm = v.region_matrix(region)
  m = vm
  for i in range(iterations):
    m = median_array(m, bin_size)

  from VolumeData import Array_Grid_Data
  d = v.data
  if v.name.endswith('median'): name = v.name
  else:                         name = '%s median' % v.name
  mg = Array_Grid_Data(m, origin, step, d.cell_angles, d.rotation,
                       name = name)
  return mg

# -----------------------------------------------------------------------------
# Volume border of result is set to zero.  Bin size must be odd.
#
def median_array(m, bin_size=3):

  if isinstance(bin_size, int):
    bin_size = (bin_size, bin_size, bin_size)
  si,sj,sk= bin_size

  from numpy import zeros, empty, median
  mm = zeros(m.shape, m.dtype)

  if m.shape[0] < sk or m.shape[1] < sj or m.shape[2] < si:
    return mm     # Size less than filter width.

  ksize, jsize, isize = m.shape
  hsi,hsj,hsk = [(n-1)/2 for n in bin_size]
  pn = empty((jsize-2*hsj,isize-2*hsi,si*sj*sk), m.dtype)

  for k in range(hsk,ksize-hsk):
    c = 0
    for ko in range(-hsk,hsk+1):
      for jo in range(-hsj,hsj+1):
        for io in range(-hsi,hsi+1):
          pn[:,:,c] = m[k+ko,hsj+jo:jsize-hsj+jo,hsi+io:isize-hsi+io]
          c += 1
    mm[k,hsj:jsize-hsj,hsi:isize-hsi] = median(pn,axis=2)

  return mm

# -----------------------------------------------------------------------------
# Test volume data set of size n containing a sphere of radius r (index units)
# with Gaussian noise.
#
def sphere_volume(n, r, noise = 1.0):

  from numpy import indices, single as floatc
  o = 0.5*(n-1)
  i = indices((n,n,n), floatc) - o
  a = (i[0,...]*i[0,...] + i[1,...]*i[1,...] + i[2,...]*i[2,...] < r*r).astype(floatc)

  if noise > 0:
    from numpy.random import normal
    a += normal(0, noise, a.shape)

  import VolumeData, VolumeViewer
  g = VolumeData.Array_Grid_Data(a, origin = (-o,-o,-o), step = (1,1,1),
                                 name = 'sphere')
  v = VolumeViewer.volume_from_grid_data(g)
  return v

def make_test_volume():
  v = sphere_volume(64, 25)
  mv = median_filter(v)
  m2v = median_filter(mv)
  m3v = median_filter(m2v)
  return m2v
