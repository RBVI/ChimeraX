# -----------------------------------------------------------------------------
# Even out density across a density map.
#
# Scale density d(x) by affine function (1 + x*u) such that the least
# squares fit hyperplane p(x) = constant.
#
# Calculation should be done in C++ for speed.
#
def flatten(volume, method = 'multiply linear',
            step = 1, subregion = None, fitregion = None,
            modelId = None, task = None):

  fg = flattened_grid(volume, method, step, subregion,
                      fitregion = fitregion, task = task)
  from VolumeViewer import volume_from_grid_data
  fv = volume_from_grid_data(fg, show_data = False, model_id = modelId)
  fv.copy_settings_from(volume, copy_region = False)
  fv.show()
  
  volume.unshow()          # Hide original map
  
  return fv

# -----------------------------------------------------------------------------
#
def flattened_grid(volume, method = 'multiply linear',
                   step = 1, subregion = None, region = None,
                   fitregion = None, task = None):

  v = volume
  if region is None:
    region = v.subregion(step, subregion)

  m = v.region_matrix(region).copy()
  if fitregion:
    fregion = v.subregion(step, fitregion)
    mfit = v.region_matrix(fregion)
    moffset = [i-fi for i,fi in zip(region[0],fregion[0])]
  else:
    mfit = m
    moffset = (0,0,0)
  flatten_matrix(m, method, mfit, moffset, task = task)

  from VolumeData import Array_Grid_Data
  d = v.data
  if v.name.endswith('flat'): name = v.name
  else:                       name = '%s flat' % v.name
  forigin, fstep = v.region_origin_and_step(region)
  fg = Array_Grid_Data(m, forigin, fstep, d.cell_angles, d.rotation,
                       name = name)
  return fg

# -----------------------------------------------------------------------------
#
def flatten_matrix(m, method = 'multiply linear',
                   mfit = None, moffset = (0,0,0), task = None):

  if mfit is None:
    mfit = m
    
  if task:
    task.updateStatus('computing moments')
  from _filter import moments, affine_scale
  v2, v1, v0 = moments(mfit)

  if method == 'multiply linear':
    # Multiply by affine function to make resulting first moments of map zero.
    # Mid-point of map is scaled by 1.0.
    u = zero_moment_scaling(v0, v1, v2, mfit.shape)
    invert = False
  elif method == 'divide linear':
    # Divide by Least squares fit of affine function scaled by a constant so
    # that mid-point of map is scaled by 1.0.
    u = least_squares_fit(v0, v1, v2, mfit.shape)
    invert = True

  if tuple(moffset) != (0,0,0):
    u = (u[0]+moffset[0]*u[1]+moffset[1]*u[2]+moffset[2]*u[3], u[1], u[2], u[3])
  # Scale by 1 at center of map.
  s0,s1,s2 = m.shape
  fc = u[0] + 0.5*s0*u[1] + 0.5*s1*u[2] + 0.5*s2*u[3]
  f = [c/fc for c in u]
  
  if task:
    task.updateStatus('scaling data')
  affine_scale(m, f[0], f[1:4], invert)

  return u

# -----------------------------------------------------------------------------
#
def zero_moment_scaling(v0, v1, v2, shape):

  s0,s1,s2 = shape
  s0sum,s1sum,s2sum = [sum_of_integers(s-1) for s in (s0,s1,s2)]
  x0 = s0*s1*s2
  x1 = (s0sum*s1*s2, s0*s1sum*s2, s0*s1*s2sum)

  A = [[(x0*v2[i][j]-x1[i]*v1[j]) for j in range(3)] for i in range(3)]
  # Avoid singular matrix for axes with only one plane.
  for a in [a for a in (0,1,2) if shape[a] == 1]:
    A[a][a] = 1
  b = [v0*x1[i] - x0*v1[i] for i in range(3)]
  from numpy.linalg import solve
  u = solve(A, b)
  return (1.0,) + tuple(u)

# -----------------------------------------------------------------------------
#
def least_squares_fit(v0, v1, v2, shape):

  s0,s1,s2 = shape
  s0sum,s1sum,s2sum = [sum_of_integers(s-1) for s in (s0,s1,s2)]
  x0 = s0*s1*s2
  x1 = [s0sum*s1*s2, s0*s1sum*s2, s0*s1*s2sum]
  x2 = [[sum_of_square_integers(s0-1)*s1*s2, s0sum*s1sum*s2, s0sum*s1*s2sum],
        [s0sum*s1sum*s2, s0*sum_of_square_integers(s1-1)*s2, s0*s1sum*s2sum],
        [s0sum*s1*s2sum, s0*s1sum*s2sum, s0*s1*sum_of_square_integers(s2-1)]]

  A = [[x0] + x1] + [ [x1[a]] + x2[a] for a in (0,1,2) ]
  # Avoid singular matrix for axes with only one plane.
  for a in [a for a in (0,1,2) if shape[a] == 1]:
    A[a+1][a+1] = 1
  b = (v0,) + tuple(v1)
  from numpy.linalg import solve
  u = solve(A, b)
  return u

# -----------------------------------------------------------------------------
#
def sum_of_integers(n):
  return (n*(n+1))/2
def sum_of_square_integers(n):
  return (n*(n+1)*(2*n+1))/6
           
# -----------------------------------------------------------------------------
#
def print_corner_scaling(f, shape):

  s0,s1,s2 = shape
  for i0,i1,i2 in ((0,0,0),(0,0,s2-1),(0,s1-1,0),(0,s1-1,s2-1),
                   (s0-1,0,0),(s0-1,0,s2-1),(s0-1,s1-1,0),(s0-1,s1-1,s2-1)):
    d = f[0] + i0*f[1] + i1*f[2] + i2*f[3]
    print (i2,i1,i0), d
