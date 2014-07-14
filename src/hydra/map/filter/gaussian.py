# -----------------------------------------------------------------------------
# Performs Gaussian convolution on current Volume Viewer data set producing
# a new data set that is shown in Volume Viewer.
#
# The standard deviation is specified in xyz units.
#
def gaussian_convolve(volume, sdev, step = 1, subregion = None,
                      value_type = None, modelId = None, task = None):

  gg = gaussian_grid(volume, sdev, step, subregion, value_type = value_type,
                     task = task)
  from VolumeViewer import volume_from_grid_data
  gv = volume_from_grid_data(gg, show_data = False, model_id = modelId)
  gv.copy_settings_from(volume, copy_region = False)
  gv.show()
  
  volume.unshow()          # Hide original map
  
  return gv

# -----------------------------------------------------------------------------
#
def gaussian_grid(volume, sdev, step = 1, subregion = None, region = None,
                  value_type = None, task = None):

  v = volume
  if region is None:
    region = v.subregion(step, subregion)

  origin, step = v.region_origin_and_step(region)

  sdev3 = (sdev,sdev,sdev) if isinstance(sdev,(float,int)) else sdev
  ijk_sdev = [float(sd)/s for sd,s in zip(sdev3,step)]

  m = v.region_matrix(region)
  gm = gaussian_convolution(m, ijk_sdev, value_type = value_type, task = task)

  from VolumeData import Array_Grid_Data
  d = v.data
  if v.name.endswith('gaussian'): name = v.name
  else:                           name = '%s gaussian' % v.name
  gg = Array_Grid_Data(gm, origin, step, d.cell_angles, d.rotation,
                       name = name)
  return gg

# -----------------------------------------------------------------------------
# Compute with zero padding in real-space to avoid cyclic-convolution.
#
def gaussian_convolution(data, ijk_sdev, value_type = None,
                         cyclic = False, cutoff = 5, task = None):

  if value_type is None:
    value_type = data.dtype

  from numpy import array, float32, float64, multiply, swapaxes
  vt = value_type if value_type == float32 or value_type == float64 else float32
  c = array(data, vt)

  from numpy.fft import rfft, irfft
  for axis in range(3):           # Transform one axis at a time.
    size = c.shape[axis]
    if size == 1:
      continue          # For a plane don't try to filter normal to plane.
    sdev = ijk_sdev[2-axis]       # Axes i,j,k are 2,1,0.
    hw = min(size/2, int(cutoff*sdev+1)) if cutoff else size/2
    nzeros = 0 if cyclic else hw  # Zero-fill for non-cyclic convolution.
    if nzeros > 0:
      # FFT performance is much better (up to 10x faster in numpy 1.2.1)
      # than other sizes.
      nzeros = efficient_fft_size(size + nzeros) - size
    g = gaussian(sdev, size + nzeros, vt)
    g[hw:-hw] = 0
    fg = rfft(g)                  # Fourier transform of 1-d gaussian.
    cs = swapaxes(c, axis, 2)     # Make axis 2 the FT axis.
    s0 = cs.shape[0]
    for p in range(s0):  # Transform one plane at a time.
      cp = cs[p,...]
      try:
        ft = rfft(cp, n=len(g))   # Complex128 result, size n/2+1
      except ValueError as e:
        raise MemoryError(e)      # Array dimensions too large.
      multiply(ft, fg, ft)
      cp[:,:] = irfft(ft)[:,:size] # Float64 result
      if task:
        pct = 100.0 * (axis + float(p)/s0) / 3.0
        task.updateStatus('%.0f%%' % pct)

  if value_type != vt:
    return c.astype(value_type)

  return c

# -----------------------------------------------------------------------------
#
def gaussian(sdev, size, value_type):

  from math import exp
  from numpy import empty, add, divide

  g = empty((size,), value_type)
  for i in range(size):
    u = min(i,size-i) / sdev
    p = min(u*u/2, 100)               # avoid OverflowError with exp()
    g[i] = exp(-p)
  area = add.reduce(g)
  divide(g, area, g)
  return g

# -----------------------------------------------------------------------------
#
def ceil_power_of_2(n):

  p = 1
  while n > p:
    p *= 2
  return p

# -----------------------------------------------------------------------------
#
fast_fft_sizes = [2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 21, 25, 32, 33, 36, 40, 48, 50, 54, 55, 64, 65, 72, 80, 81, 90, 96, 100, 108, 120, 128, 144, 150, 160, 162, 180, 192, 200, 216, 256, 270, 300, 320, 324, 325, 336, 360, 400, 432, 450, 480, 500, 512, 540, 576, 600, 640, 648, 720, 750, 800, 810, 864, 900, 960, 1024, 1080, 1152, 1200, 1280, 1296, 1350, 1440, 1500, 1536, 1600, 1620, 1728, 1800, 1920, 2048, 2160, 2250, 2304, 2400, 2560, 2700, 2880, 3000, 3072, 3200, 3240, 3600, 3840, 4096, 4160, 4176, 4180, 4186, 4187, 4192]

def efficient_fft_size(n):

  if n < fast_fft_sizes[-1]:
    from bisect import bisect
    b = bisect(fast_fft_sizes, n)
    s = fast_fft_sizes[b]
  else:
    s = ceil_power_of_2(n)
  return s

# -----------------------------------------------------------------------------
# Find array sizes where computing FFT with numpy is fast.
# This code is for determining the table usd by efficient_fft_size().
#
# Typically power of 2 sizes are fastest, sizes with all small prime factors
# are usually fast, and others can be much slower, up to 10x slower.
# Results depend on FFT recursion algorithm and CPU cache performance.
# Different CPUs will give different fastest sizes.
#
# Specifically find a list of sizes that excludes sizes that are slower than
# some larger size.  This is useful when we can zero pad the array to any
# amount for example in Gaussian filtering.
#
def find_fast_fft_sizes(max_size = 4192, array_size = 1024*1024):

    from numpy.fft import rfft, irfft
    from time import clock
    from numpy import ones, single
    a = ones((array_size,), single)
    tl = []
    for s in range(2, max_size+1):
        n = array_size/s
        b = a[:s*n].reshape((n, s))
        c0 = clock()
        f = rfft(b)
        g = irfft(f)
        c1 = clock()
        t = (c1-c0)/n           # Average time for one fft
        if t < 0:
          # Python clock() wraps after 4295 seconds on 32-bit systems. Ugh.
          t += (256.0**4/1e6)/n
        tl.append((t,s))
        print (s, '%.4g' % (1e6*t))
    tl.reverse()
    tmin,s = tl[0]
    fs = [s]
    for t,s in tl:
        if t < tmin:
            fs.append(s)
            tmin = t
    fs.reverse()
    return fs
