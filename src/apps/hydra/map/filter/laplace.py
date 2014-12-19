# -----------------------------------------------------------------------------
# Take Laplacian of current Volume Viewer data set producing a new data set
# that is shown in Volume Viewer.  Laplacian
#
#   L = (d/dx)^2 + (d/dy)^2 + (d/dz)^2
#
# is approaximated with "-1 2 -1" kernel in each dimension.  Resulting volume
# is same size with edge voxels set to zero.
#
def laplacian(v = None, step = None, subregion = None, model_id = None):

  if v is None:
    from VolumeViewer import active_volume
    v = active_volume()
    if v is None:
      return None

  m = v.matrix(step = step, subregion = subregion)

  from numpy import float32, multiply, add
  lm = m.astype(float32)        # Copy array
  multiply(lm, -6.0, lm)
  add(lm[:-1,:,:], m[1:,:,:], lm[:-1,:,:])
  add(lm[1:,:,:], m[:-1,:,:], lm[1:,:,:])
  add(lm[:,:-1,:], m[:,1:,:], lm[:,:-1,:])
  add(lm[:,1:,:], m[:,:-1,:], lm[:,1:,:])
  add(lm[:,:,:-1], m[:,:,1:], lm[:,:,:-1])
  add(lm[:,:,1:], m[:,:,:-1], lm[:,:,1:])

  lm[0,:,:] = 0
  lm[-1,:,:] = 0
  lm[:,0,:] = 0
  lm[:,-1,:] = 0
  lm[:,:,0] = 0
  lm[:,:,-1] = 0

  origin, step = v.data_origin_and_step(subregion = subregion, step = step)
  d = v.data
  from VolumeData import Array_Grid_Data
  ld = Array_Grid_Data(lm, origin, step, d.cell_angles, d.rotation,
                       name = v.name + ' Laplacian')
  ld.polar_values = True
  from VolumeViewer import volume_from_grid_data
  lv = volume_from_grid_data(ld, show_data = False, model_id = model_id)
  lv.copy_settings_from(v, copy_thresholds = False, copy_colors = False)
  lv.set_parameters(cap_faces = False)
  lv.initialize_thresholds()
  lv.show()
  
  v.unshow()          # Hide original map

  return lv
