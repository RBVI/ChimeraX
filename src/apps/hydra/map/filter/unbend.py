# -----------------------------------------------------------------------------
# Straighten the volume around a path.  Used to straighten a long helical
# bacteria so slices can easily show interior structure along full length.
#
# path is list of points in global coordinates.
# yaxis is vector (3-tuple) in global coordinates.
# xsize, ysze, grid_spacing are in physical units.
#
def unbend_volume(volume, path, yaxis, xsize, ysize, grid_spacing,
                  subregion = 'all', step = 1, model_id = None, open = True):

  # Compute correctly spaced cubic splined path points.
  points = spline_path(path, grid_spacing)
  axes = path_point_axes(points, yaxis)
  nx = int(xsize/grid_spacing) + 1
  ny = int(ysize/grid_spacing) + 1
  nz = len(points)

  # Create a rectangle of point positions to interpolate at.
  from numpy import empty, float32, arange
  section = empty((ny,nx,3), float32)
  x = arange(nx,dtype=float32)*grid_spacing - 0.5*(xsize-1.0)
  y = arange(ny,dtype=float32)*grid_spacing - 0.5*(ysize-1.0)
  for j in range(ny):
    section[j,:,0] = x
  for i in range(nx):
    section[:,i,1] = y
  section[:,:,2] = 0
  s = section.reshape((ny*nx,3))

  # Interpolate planes to fill straightened array.
  from Matrix import chimera_xform
  m = empty((nz,ny,nx), float32)
  for k in range(nz):
    a = axes[k]
    p = points[k]
    tf = [([a[c][r] for c in (0,1,2)]+[p[r]]) for r in (0,1,2)]
    xf = chimera_xform(tf)
    m[k,:,:] = volume.interpolated_values(s, xf, subregion=subregion, step=step).reshape((ny,nx))

  # Create volume.
  from VolumeData import Array_Grid_Data
  step = [grid_spacing] * 3
  origin = [0,0,0]
  g = Array_Grid_Data(m, origin, step, name = 'unbend')
  from VolumeViewer import volume_from_grid_data
  v = volume_from_grid_data(g, show_data = False, model_id = model_id,
                            open_model = open, show_dialog = open)
  v.copy_settings_from(volume, copy_region = False, copy_active = False,
                       copy_xform = open)
  if open:
    v.show()

  return v

# -----------------------------------------------------------------------------
#
def spline_path(path, grid_spacing):

  oversample = 3
  from Matrix import distance
  subdiv = [int(oversample * distance(path[i+1], path[i]) / grid_spacing)
            for i in range(len(path)-1)]
  from numpy import array
  npath = array(path)
  from VolumePath import spline
  points = spline.natural_cubic_spline(npath, subdiv)
  epoints = equispaced_points(points, grid_spacing)
  return epoints

# -----------------------------------------------------------------------------
#
def equispaced_points(points, grid_spacing):

  ep = [points[0]]
  from VolumePath import spline
  arcs = spline.arc_lengths(points)
  d = grid_spacing
  from Matrix import linear_combination
  for i, a in enumerate(arcs):
    while a > d:
      f = (d - arcs[i-1]) / (arcs[i] - arcs[i-1])
      ep.append(linear_combination((1-f), points[i-1], f, points[i]))
      d += grid_spacing
  return ep
  
# -----------------------------------------------------------------------------
#
def path_point_axes(points, yaxis):

  zaxes = path_tangents(points)
  from Matrix import orthonormal_frame
  axes = [orthonormal_frame(za) for za in zaxes]
  return axes
  
# -----------------------------------------------------------------------------
#
def path_tangents(points):

  # TODO: Handle coincident points.
  from Matrix import linear_combination, normalize_vector
  tang = [linear_combination(1, points[1], -1, points[0])]
  for i in range(1,len(points)-1):
    tang.append(linear_combination(1, points[i+1], -1, points[i-1]))
  tang.append(linear_combination(1, points[-1], -1, points[-2]))
  ntang = [normalize_vector(t) for t in tang]
  return ntang

# -----------------------------------------------------------------------------
#
def atom_path(atoms):
  
  from VolumePath.tube import atom_chains
  chains = atom_chains(atoms)
  if len(chains) != 1:
    raise RuntimeError('Require 1 chain of atoms, got %d' % len(chains))
  chain = chains[0][0]
  points = [a.xformCoord().data() for a in chain]
  return points

# -----------------------------------------------------------------------------
# Use active volume and path defined by selected atoms to unbend volume.
#
def test_unbend(yaxis = (0,0,1), isize = 20, jsize = 20):

  from chimera.selection import currentAtoms
  atoms = currentAtoms()
  path = atom_path(atoms)
  from VolumeViewer import active_volume
  v = active_volume()
  grid_spacing = min(v.data.step)
  xsize = isize * grid_spacing
  ysize = jsize * grid_spacing
  unbend_volume(v, path, yaxis, xsize, ysize, grid_spacing)
