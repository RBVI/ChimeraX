# -----------------------------------------------------------------------------
# Find map grid points that are a maximum along 2 of the 3 axes.
#
def ridges(volume, level = None, step = 1, subregion = None, modelId = None):

    if level is None:
        level = min(volume.surface_levels)
    rg = ridge_grid(volume, level, step, subregion)
    from VolumeViewer import volume_from_grid_data
    rv = volume_from_grid_data(rg, model_id = modelId)
    rv.set_parameters(surface_levels = [0.5])
    rv.show()
    volume.unshow()          # Hide original map
  
    return rv

# -----------------------------------------------------------------------------
#
def ridge_grid(volume, level = None, step = 1, subregion = None):

  v = volume
  region = v.subregion(step, subregion)

  m = v.region_matrix(region)
  rm = ridge_matrix(m, level)

  from VolumeData import Array_Grid_Data
  d = v.data
  name = '%s ridges' % v.name
  origin, step = v.region_origin_and_step(region)
  rg = Array_Grid_Data(rm, origin, step, d.cell_angles, d.rotation,
                       name = name)
  return rg

# -----------------------------------------------------------------------------
#
def ridge_matrix(m, level = None):
    return ridge_matrix_all(m, level)
#    return ridge_matrix_edge(m, level)

# -----------------------------------------------------------------------------
#
def ridge_matrix_all(m, level = None):

    from numpy import zeros, int8, logical_and, greater_equal
    rm = zeros(m.shape, int8)
    rmm = rm[1:-1,1:-1,1:-1]
    mm = m[1:-1,1:-1,1:-1]
    # 6 edges
    rmm += logical_and(mm > m[1:-1,:-2,:-2], mm > m[1:-1,2:,2:])
    rmm += logical_and(mm > m[1:-1,2:,:-2], mm > m[1:-1,:-2,2:])
    rmm += logical_and(mm > m[2:,1:-1,2:], mm > m[:-2,1:-1,:-2])
    rmm += logical_and(mm > m[2:,1:-1,:-2], mm > m[:-2,1:-1,2:])
    rmm += logical_and(mm > m[2:,2:,1:-1], mm > m[:-2,:-2,1:-1])
    rmm += logical_and(mm > m[2:,:-2,1:-1], mm > m[:-2,2:,1:-1])
    # 4 diagonals
    rmm += logical_and(mm > m[2:,:-2,:-2], mm > m[:-2,2:,2:])
    rmm += logical_and(mm > m[2:,:-2,2:], mm > m[:-2,2:,:-2])
    rmm += logical_and(mm > m[2:,2:,:-2], mm > m[:-2,:-2,2:])
    rmm += logical_and(mm > m[2:,2:,2:], mm > m[:-2,:-2,:-2])
    # 3 faces
    rmm += logical_and(mm > m[1:-1,1:-1,:-2], mm > m[1:-1,1:-1,2:])
    rmm += logical_and(mm > m[1:-1,:-2,1:-1], mm > m[1:-1,2:,1:-1])
    rmm += logical_and(mm > m[:-2,1:-1,1:-1], mm > m[2:,1:-1,1:-1])

#    greater_equal(rm, 4, rm)
    if not level is None:
        rm[m < level] = 0
#        logical_and(rm, m >= level, rm)
    
    return rm

# -----------------------------------------------------------------------------
#
def ridge_matrix_edge(m, level = None):

    from numpy import zeros, int8, logical_and, greater_equal
    rm = zeros(m.shape, int8)
    rmm = rm[1:-1,1:-1,1:-1]
    mm = m[1:-1,1:-1,1:-1]
    rmm += logical_and(mm > m[1:-1,:-2,:-2], mm > m[1:-1,2:,2:])
    rmm += logical_and(mm > m[1:-1,2:,:-2], mm > m[1:-1,:-2,2:])

    rmm += logical_and(mm > m[2:,1:-1,2:], mm > m[:-2,1:-1,:-2])
    rmm += logical_and(mm > m[2:,1:-1,:-2], mm > m[:-2,1:-1,2:])

    rmm += logical_and(mm > m[2:,2:,1:-1], mm > m[:-2,:-2,1:-1])
    rmm += logical_and(mm > m[2:,:-2,1:-1], mm > m[:-2,2:,1:-1])

    greater_equal(rm, 4, rm)
    if not level is None:
        logical_and(rm, m >= level, rm)
    
    return rm

# -----------------------------------------------------------------------------
#
def ridge_matrix_diag(m, level = None):

    from numpy import zeros, int8, logical_and, greater_equal
    rm = zeros(m.shape, int8)
    rmm = rm[1:-1,1:-1,1:-1]
    mm = m[1:-1,1:-1,1:-1]
    rmm += logical_and(mm > m[2:,:-2,:-2], mm > m[:-2,2:,2:])
    rmm += logical_and(mm > m[2:,:-2,2:], mm > m[:-2,2:,:-2])
    rmm += logical_and(mm > m[2:,2:,:-2], mm > m[:-2,:-2,2:])
    rmm += logical_and(mm > m[2:,2:,2:], mm > m[:-2,:-2,:-2])
#    greater_equal(rm, 2, rm)
    greater_equal(rm, 3, rm)
    if not level is None:
        logical_and(rm, m >= level, rm)
    
    return rm

# -----------------------------------------------------------------------------
#
def ridge_matrix_face(m, level = None):

    from numpy import zeros, int8, logical_and, greater_equal
    rm = zeros(m.shape, int8)
    rmm = rm[1:-1,1:-1,1:-1]
    mm = m[1:-1,1:-1,1:-1]
    rmm += logical_and(mm > m[1:-1,1:-1,:-2], mm > m[1:-1,1:-1,2:])
    rmm += logical_and(mm > m[1:-1,:-2,1:-1], mm > m[1:-1,2:,1:-1])
    rmm += logical_and(mm > m[:-2,1:-1,1:-1], mm > m[2:,1:-1,1:-1])

    greater_equal(rm, 2, rm)
    if not level is None:
        logical_and(rm, m >= level, rm)
    
    return rm

# -----------------------------------------------------------------------------
#
def ridge_matrix_face2(m, level = None):

    from numpy import zeros, int8, logical_and, greater_equal
    rm = zeros(m.shape, int8)
    ksize, jsize, isize = m.shape
    for k in range(1,ksize-1):
        mk = m[k,:,:]
        rm[k,:,:] += logical_and(mk > m[k-1,:,:], mk > m[k+1,:,:])
    for j in range(1,jsize-1):
        mj = m[:,j,:]
        rm[:,j,:] += logical_and(mj > m[:,j-1,:], mj > m[:,j+1,:])
    for i in range(1,isize-1):
        mi = m[:,:,i]
        rm[:,:,i] += logical_and(mi > m[:,:,i-1], mi > m[:,:,i+1])
    greater_equal(rm, 2, rm)
    if not level is None:
        logical_and(rm, m >= level, rm)
    
    return rm
