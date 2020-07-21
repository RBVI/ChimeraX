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
# Take Laplacian of current Volume Viewer data set producing a new data set
# that is shown in Volume Viewer.  Laplacian
#
#   L = (d/dx)^2 + (d/dy)^2 + (d/dz)^2
#
# is approaximated with "-1 2 -1" kernel in each dimension.  Resulting volume
# is same size with edge voxels set to zero.
#
def laplacian(v, step = None, subregion = None, model_id = None):

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
  from chimerax.map_data import ArrayGridData
  ld = ArrayGridData(lm, origin, step, d.cell_angles, d.rotation,
                       name = v.name + ' Laplacian')
  ld.polar_values = True
  from chimerax.map import volume_from_grid_data
  lv = volume_from_grid_data(ld, v.session, model_id = model_id)
  lv.copy_settings_from(v, copy_thresholds = False, copy_colors = False)
  lv.set_parameters(cap_faces = False)
  
  v.display = False          # Hide original map

  return lv
