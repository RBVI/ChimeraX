# -----------------------------------------------------------------------------
# The Grid_Data class defines the data format needed by the Chimera
# Volume Viewer extension to produce surface, mesh, and solid
# displays.  It should be used as a base class for reading specific 3D data
# formats, for example, Delphi electrostatic potentials, DSN6 electron density
# maps, MRC electron microscope density maps, and Priism 3D microscope images.
#
# The simplest volume data is just a 3 dimensional matrix of values.
# This could be kept in memory as a 3 dimensional NumPy array.
# The Grid_Data object is an extension of this case in a couple ways.
#
# Grid_Data defines how the data is positioned in an xyz coordinate space.
# Each data value is thought of as a sample at a particular point in space.
# The data value with index (0,0,0) has xyz postion given by origin.
# The step parameter (3 values) gives the spacing between data values
# along the 3 data axes. The cell_angles = (a,b,c) describe the
# skewing of the data axes where a is the angle between the y and z axes,
# b is the angle between x and z axes, and c is the angle between x and y
# axes.  The angles are measured in degrees.
#
from numpy import float32
class Grid_Data:

  def __init__(self, size,
               value_type = float32,
               origin = (0,0,0),
               step = (1,1,1),
               cell_angles = (90,90,90),
               rotation = ((1,0,0),(0,1,0),(0,0,1)),
               symmetries = (),
               name = '',
               path = '',       # Can be list of paths
               file_type = '',
               grid_id = '',
               default_color = (.7,.7,.7,1)):

    # Path, file_type and grid_id are for reloading data sets.
    self.path = path
    self.file_type = file_type  # 'mrc', 'spider', ....
    self.grid_id = grid_id      # String identifying grid in multi-grid files.
    
    if name == '':
      name = self.name_from_path(path)
    self.name = name
    
    self.size = tuple(size)

    from numpy import dtype
    if not isinstance(value_type, dtype):
      value_type = dtype(value_type)
    self.value_type = value_type        # numpy dtype.

    # Parameters defining how data matrix is positioned in space
    self.origin = tuple(origin)
    self.original_origin = self.origin
    self.step = tuple(step)
    self.original_step = self.step
    self.cell_angles = tuple(cell_angles)
    self.rotation = tuple(map(tuple, rotation))
    self.symmetries = symmetries
    self.ijk_to_xyz_transform = None
    self.xyz_to_ijk_transform = None

    self.rgba = default_color            # preferred color for displaying data

    global data_cache
    self.data_cache = data_cache

    self.writable = False
    self.change_callbacks = []

    self.update_transform()

  # ---------------------------------------------------------------------------
  #
  def set_path(self, path, format = None):

    if path != self.path:
      self.path = path
      self.name = self.name_from_path(path)
      self.call_callbacks('path changed')
      
    if format and format != self.file_type:
      self.file_type = format
      self.call_callbacks('file format changed')

  # ---------------------------------------------------------------------------
  #
  def name_from_path(self, path):

    from os.path import basename
    if isinstance(path, (list,tuple)):  p = path[0]
    else:                               p = path
    name = basename(p)
    return name

  # ---------------------------------------------------------------------------
  #
  def set_origin(self, origin):

    if tuple(origin) != self.origin:
      self.origin = tuple(origin)
      self.update_transform()

    # TODO: Update symmetries for origin, step, cell angle and rotation changes

  # ---------------------------------------------------------------------------
  #
  def set_step(self, step):

    if tuple(step) != self.step:
      self.step = tuple(step)
      self.update_transform()

  # ---------------------------------------------------------------------------
  #
  def set_cell_angles(self, cell_angles):

    if tuple(cell_angles) != self.cell_angles:
      self.cell_angles = tuple(cell_angles)
      self.update_transform()

  # ---------------------------------------------------------------------------
  #
  def set_rotation(self, rotation):

    r = tuple(map(tuple, rotation))
    if r != self.rotation:
      self.rotation = r
      self.update_transform()

  # ---------------------------------------------------------------------------
  # Compute 3 by 4 matrices encoding rotation and translation.
  #
  def update_transform(self):

    from .. import place
    saxes = place.skew_axes(self.cell_angles).axes()
    r = place.Place()
    r.matrix[:,:3] = self.rotation
    rsaxes = r * saxes
    tf, tf_inv = transformation_and_inverse(self.origin, self.step, rsaxes)
    if (self.ijk_to_xyz_transform is None or not tf.same(self.ijk_to_xyz_transform) or
        self.xyz_to_ijk_transform is None or not tf_inv.same(self.xyz_to_ijk_transform)):
      self.ijk_to_xyz_transform = tf
      self.xyz_to_ijk_transform = tf_inv
      self.coordinates_changed()

  # ---------------------------------------------------------------------------
  # A matrix ijk corresponds to a point in xyz space.
  # This function maps the xyz point to the matrix index.
  # The returned matrix index need not be integers.
  #
  def xyz_to_ijk(self, xyz):

    return self.xyz_to_ijk_transform * xyz

  # ---------------------------------------------------------------------------
  # A matrix ijk corresponds to a point in xyz space.
  # This function maps the matrix index to the xyz point.
  #
  def ijk_to_xyz(self, ijk):

    return self.ijk_to_xyz_transform * ijk
    
  # ---------------------------------------------------------------------------
  # Spacings in xyz space of jk, ik, and ij planes.
  #
  def plane_spacings(self):

    spacings = map(lambda u: 1.0/norm(u[:3]), self.xyz_to_ijk_transform)
    return spacings
    
  # ---------------------------------------------------------------------------
  #
  def matrix(self, ijk_origin = (0,0,0), ijk_size = None,
             ijk_step = (1,1,1), progress = None, from_cache_only = False):

    if ijk_size == None:
      ijk_size = self.size

    m = self.cached_data(ijk_origin, ijk_size, ijk_step)
    if m is None and not from_cache_only:
      m = self.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
      self.cache_data(m, ijk_origin, ijk_size, ijk_step)

    return m
    
  # ---------------------------------------------------------------------------
  # Must overide this function in derived class to return a 3 dimensional
  # NumPy matrix.  The returned matrix has size ijk_size and
  # element ijk is accessed as m[k,j,i].  It is an error if the requested
  # submatrix does not lie completely within the full data matrix.  It is
  # also an error for the size to be <= 0 in any dimension.  These invalid
  # inputs might throw an exception or might return garbage.  It is the
  # callers responsibility to make sure the arguments are valid.
  #
  def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):

    raise NotImplementedError('Grid %s has no read_matrix() routine' % self.name)
  
  # ---------------------------------------------------------------------------
  # Convenience routine.
  #
  def matrix_slice(self, matrix, ijk_origin, ijk_size, ijk_step):

    i1, j1, k1 = ijk_origin
    i2, j2, k2 = map(lambda i, s: i+s, ijk_origin, ijk_size)
    istep, jstep, kstep = ijk_step
    m = matrix[k1:k2:kstep, j1:j2:jstep, i1:i2:istep]
    return m
    
  # ---------------------------------------------------------------------------
  # Deprecated.  Used before matrix() routine existed.
  #
  def full_matrix(self, progress = None):

    matrix = self.matrix()
    return matrix
    
  # ---------------------------------------------------------------------------
  # Deprecated.  Used before matrix() routine existed.
  #
  def submatrix(self, ijk_origin, ijk_size):

    return self.matrix(ijk_origin, ijk_size)

  # ---------------------------------------------------------------------------
  #
  def cached_data(self, origin, size, step):

    dcache = self.data_cache
    if dcache is None:
      return None

    key = (self, tuple(origin), tuple(size), tuple(step))
    m = dcache.lookup_data(key)
    if not m is None:
      return m

    # Look for a matrix containing the desired matrix
    group = self
    kd = dcache.group_keys_and_data(group)
    for k, matrix in kd:
      orig, sz, st = k[1:]
      if (step[0] < st[0] or step[1] < st[1] or step[2] < st[2] or
          step[0] % st[0] or step[1] % st[1] or step[2] % st[2]):
        continue        # Step sizes not compatible
      if (origin[0] < orig[0] or origin[1] < orig[1] or origin[2] < orig[2] or
          origin[0] + size[0] > orig[0] + sz[0] or
          origin[1] + size[1] > orig[1] + sz[1] or
          origin[2] + size[2] > orig[2] + sz[2]):
        continue        # Doesn't cover.
      dstep = [a//b for a,b in zip(step, st)]
      offset = [a-b for a,b in zip(origin, orig)]
      if offset[0] % st[0] or offset[1] % st[1] or offset[2] % st[2]:
        continue        # Offset stagger.
      moffset = [o // s for o,s in zip(offset, st)]
      msize = [(s+t-1) // t for s,t in zip(size, st)]
      m = matrix[moffset[2]:moffset[2]+msize[2]:dstep[2],
                 moffset[1]:moffset[1]+msize[1]:dstep[1],
                 moffset[0]:moffset[0]+msize[0]:dstep[0]]
      dcache.lookup_data(key)			# update access time
      return m

    return None

  # ---------------------------------------------------------------------------
  #
  def cache_data(self, m, origin, size, step):

    dcache = self.data_cache
    if dcache is None:
      return

    key = (self, tuple(origin), tuple(size), tuple(step))
    elements = m.size
    bytes = elements * m.itemsize
    groups = [self]
    descrip = self.data_description(origin, size, step)
    dcache.cache_data(key, m, bytes, descrip, groups)

  # ---------------------------------------------------------------------------
  #
  def data_description(self, origin, size, step):

    description = self.name

    if origin == (0,0,0):
      bounds = ' (%d,%d,%d)' % tuple(size)
    else:
      region = (origin[0], origin[0]+size[0]-1,
		origin[1], origin[1]+size[1]-1,
		origin[2], origin[2]+size[2]-1)
      bounds = ' (%d-%d,%d-%d,%d-%d)' % region
    description += bounds

    if step != (1,1,1):
      description += ' step (%d,%d,%d)' % tuple(step)

    return description

  # ---------------------------------------------------------------------------
  #
  def clear_cache(self):

    dcache = self.data_cache
    if dcache is None:
      return

    for k,d in dcache.group_keys_and_data(self):
      dcache.remove_key(k)

  # ---------------------------------------------------------------------------
  #
  def add_change_callback(self, cb):

    self.change_callbacks.append(cb)

  # ---------------------------------------------------------------------------
  #
  def remove_change_callback(self, cb):

    self.change_callbacks.remove(cb)

  # ---------------------------------------------------------------------------
  # Code has modified matrix elements, or the value type has changed.
  #
  def values_changed(self):

    self.call_callbacks('values changed')

  # ---------------------------------------------------------------------------
  # Mapping of array indices to xyz coordinates has changed.
  #
  def coordinates_changed(self):

    self.call_callbacks('coordinates changed')

  # ---------------------------------------------------------------------------
  #
  def call_callbacks(self, reason):
    
    for cb in self.change_callbacks:
      cb(reason)
    
# -----------------------------------------------------------------------------
# Return 3 by 4 matrix where first 3 columns give rotation and last column
# is translation.
#
def transformation_and_inverse(origin, step, axes):
  
  ox, oy, oz = origin
  d0, d1, d2 = step
  ax, ay, az = axes

  from ..place import Place
  tf = Place(((d0*ax[0], d1*ay[0], d2*az[0], ox),
              (d0*ax[1], d1*ay[1], d2*az[1], oy),
              (d0*ax[2], d1*ay[2], d2*az[2], oz)))
  tf_inv = tf.inverse()
  
  return tf, tf_inv

# -----------------------------------------------------------------------------
# Apply scaling and skewing transformations.
#
def scale_and_skew(ijk, step, cell_angles):

  from .. import place
  import numpy
  xyz = place.skew_axes(cell_angles) * (numpy.array(step) * ijk)
  return xyz
    
# -----------------------------------------------------------------------------
#
def apply_rotation(r, v):
  
  rv = [r[a][0]*v[0] + r[a][1]*v[1] + r[a][2]*v[2] for a in (0,1,2)]
  return tuple(rv)

# -----------------------------------------------------------------------------
#
class Grid_Subregion(Grid_Data):

  def __init__(self, grid_data, ijk_min, ijk_max, ijk_step = (1,1,1)):

    self.full_data = grid_data

    ijk_min = map(lambda a,s: ((a+s-1)/s)*s, ijk_min, ijk_step)
    self.ijk_offset = ijk_min
    self.ijk_step = ijk_step

    size = map(lambda a,b,s: max(0,(b-a+s)/s), ijk_min, ijk_max, ijk_step)
    origin = grid_data.ijk_to_xyz(ijk_min)
    step = [ijk_step[a]*grid_data.step[a] for a in range(3)]

    Grid_Data.__init__(self, size, grid_data.value_type,
                       origin, step, grid_data.cell_angles,
                       grid_data.rotation, grid_data.symmetries,
                       name = grid_data.name + ' subregion')
    self.rgba = grid_data.rgba
    self.data_cache = None      # Caching done by underlying grid.
        
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    origin, step, size = self.full_region(ijk_origin, ijk_size, ijk_step)
    m = self.full_data.matrix(origin, size, step, progress)
    return m
        
  # ---------------------------------------------------------------------------
  #
  def cached_data(self, ijk_origin, ijk_size, ijk_step):

    origin, step, size = self.full_region(ijk_origin, ijk_size, ijk_step)
    m = self.full_data.cached_data(origin, size, step)
    return m
        
  # ---------------------------------------------------------------------------
  #
  def full_region(self, ijk_origin, ijk_size, ijk_step):

    origin = map(lambda i,s,o: i*s+o,ijk_origin, self.ijk_step, self.ijk_offset)
    size = map(lambda a,b: a*b, ijk_size, self.ijk_step)
    step = map(lambda a,b: a*b, ijk_step, self.ijk_step)
    return origin, step, size

  # ---------------------------------------------------------------------------
  #
  def clear_cache(self):

    self.full_data.clear_cache()

# -----------------------------------------------------------------------------
#
def norm(v):

  from math import sqrt
  d = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
  return d

# -----------------------------------------------------------------------------
# Default data cache.
#
from . import datacache
data_cache = datacache.Data_Cache(size = 0)
