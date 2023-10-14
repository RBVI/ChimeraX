# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# The GridData class defines the data format needed by the ChimeraX
# Volume Viewer extension to produce surface, mesh, and image rendering
# displays.  It should be used as a base class for reading specific 3D data
# formats, for example, Delphi electrostatic potentials, DSN6 electron density
# maps, MRC electron microscope density maps, and Priism 3D microscope images.
#
# The simplest volume data is just a 3 dimensional matrix of values.
# This could be kept in memory as a 3 dimensional NumPy array.
# The GridData object is an extension of this case in a couple ways.
#
# GridData defines how the data is positioned in an xyz coordinate space.
# Each data value is thought of as a sample at a particular point in space.
# The data value with index (0,0,0) has xyz postion given by origin.
# The step parameter (3 values) gives the spacing between data values
# along the 3 data axes. The cell_angles = (a,b,c) describe the
# skewing of the data axes where a is the angle between the y and z axes,
# b is the angle between x and z axes, and c is the angle between x and y
# axes.  The angles are measured in degrees.
#
from numpy import float32
class GridData:
  '''
  3-dimensional array of numeric values usually representing a density map
  from electron microscopy, x-ray crystallography or optical imaging.
  The grid points are positioned in space with array index (0,0,0) at
  the xyz origin, and spacing along the xyz axes given by the step parameter.
  The rectangular grid can be skewed by specifying angles between each pair
  of axes as is done to describe crystallographic unit cells.  The grid
  can also be rotated.  Values are read from a file for rectangular subregions
  of the data and cached in memory.  The numeric values can be signed or unsigned
  integer of 8, 16, or 32 bits or real 32-bit or 64-bit values.

  Attributes
  ----------
  size : 3 integers
    The number of grid points along x,y,z axes.
  value_type : numpy.dtype
    Numeric type of the data values.  Default numpy.float32
  origin : 3 floats
    Position of grid index (0,0,0) in physical coordinates (x,y,z) (usually Angstroms).
    Default (0,0,0).
  step : 3 floats
    Grid plane spacing along x,y,z axes.  Default (1,1,1)
  cell_angles : 3 floats
    Cell angles for skewed crystallography maps.
    Angles (alpha,beta,gamma) between yz, xz, and xy axes in degrees.
    Default (90,90,90).
  rotation : 3x3 matrix
    Rotation matrix in physical coordinates. Default ((1,0,0),(0,1,0),(0,0,1))
  symmetries : :class:`~.chimerax.geometry.Places` or None
    Symmetry transformations that map the to itself in physical coordinates.
    Default None means no symmetries.
  name : string
    Descriptive name of grid data.  Default ''
  path : string or list of strings
    File path or paths to location of data file.  Default ''.
    Data need not come from a file.
  file_type : string
    File format name when data is read from a file.  Default ''.
  grid_id : string
    Identifier for a specific data set in a file for file formats that
    allow more than one map in a single file, such as HDF5 format.
  default_color : 4 floats or None
    Red, green, blue, alpha color values (range 0-1) indicating preferred color
  time : integer or None
    Integer time point for time series volume data.  Usually 0 based but not required.
    Default None.
  channel : integer or None
    Integer channel number for multi-channel data (e.g. light microscopy).
    Default None.
  '''
  def __init__(self, size,
               value_type = float32,
               origin = (0,0,0),
               step = (1,1,1),
               cell_angles = (90,90,90),
               rotation = ((1,0,0),(0,1,0),(0,0,1)),
               symmetries = None,
               name = '',
               path = '',       # Can be list of paths
               file_type = '',
               grid_id = '',
               default_color = None,
               time = None,
               channel = None):
    '''Supported API.'''
    
    # Path, file_type and grid_id are for reloading data sets.
    self.path = path
    self.file_type = file_type  # 'mrc', 'spider', ....
    self.grid_id = grid_id      # String identifying grid in multi-grid files.
    
    if name == '':
      name = self.name_from_path(path)
    self.name = str(name)
    
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
    self.rotation = tuple(tuple(row) for row in rotation)
    self.symmetries = symmetries	# Places object or None
    self.ijk_to_xyz_transform = None
    self.xyz_to_ijk_transform = None

    self.rgba = default_color            # preferred color for displaying data

    self.time = time			# Integer, position in time series
    self.channel = channel		# Integer, channel number for multi-channel data

    self.data_cache = None

    self.writable = False
    self.change_callbacks = []

    self.update_transform()


  # Intended to be abstract; should be defined in subclasses of GridData
  def show_info(self):
    pass

  # ---------------------------------------------------------------------------
  # Return dictionary of options for GridData constructor used to initialize
  # a new GridData.  Keyword options replace the specified settings.
  #
  def settings(self, **replace):
    attrs = ('size', 'value_type', 'origin', 'step', 'cell_angles', 'rotation', 'symmetries',
             'name', 'time', 'channel')
    s = {attr:getattr(self, attr) for attr in attrs}
    s['default_color'] = self.rgba
    for k in replace.keys():
      if k not in attrs:
        raise ValueError('Unknown argument to GridData settings(): "%s"' % k)
    s.update(replace)
    return s
      
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

    r = tuple(tuple(row) for row in rotation)
    if r != self.rotation:
      self.rotation = r
      self.update_transform()

  # ---------------------------------------------------------------------------
  # Compute 3 by 4 matrices encoding rotation and translation.
  #
  def update_transform(self):

    from chimerax.geometry import place
    saxes = place.skew_axes(self.cell_angles).axes()
    from numpy import transpose
    r = place.Place(axes = transpose(self.rotation))
    rsaxes = r * saxes
    tf, tf_inv = transformation_and_inverse(self.origin, self.step, rsaxes)
    if (self.ijk_to_xyz_transform is None or not tf.same(self.ijk_to_xyz_transform) or
        self.xyz_to_ijk_transform is None or not tf_inv.same(self.xyz_to_ijk_transform)):
      self.ijk_to_xyz_transform = tf
      self.xyz_to_ijk_transform = tf_inv
      self.coordinates_changed()

  # ---------------------------------------------------------------------------
  #
  def xyz_to_ijk(self, xyz):
    '''
    Supported API.
    A matrix i,j,k index corresponds to a point in x,y,z space.
    This function maps the xyz point to the matrix index.
    The returned matrix index is floating point and need not be integers.
    '''
    return self.xyz_to_ijk_transform * xyz

  # ---------------------------------------------------------------------------
  #
  def ijk_to_xyz(self, ijk):
    '''
    Supported API.
    A matrix i,j,k index corresponds to a point in x,y,z space.
    This function maps the matrix index to the xyz point.
    The index can be floating point, non-integral values.
    '''

    return self.ijk_to_xyz_transform * ijk
    
  # ---------------------------------------------------------------------------
  #
  def voxel_volume(self):
    '''
    Supported API.
    Volume of one voxel including skewing.
    '''
    return self.ijk_to_xyz_transform.determinant()
    
  # ---------------------------------------------------------------------------
  #
  def voxel_count(self):
    '''
    Supported API.
    Return the total number of voxels.
    '''
    s = self.size
    return s[0]*s[1]*s[2]
    
  # ---------------------------------------------------------------------------
  # Spacings in xyz space of jk, ik, and ij planes.
  #
  def plane_spacings(self):

    spacings = [1.0/norm(u[:3]) for u in self.xyz_to_ijk_transform.matrix]
    return spacings
    
  # ---------------------------------------------------------------------------
  #
  def matrix(self, ijk_origin = (0,0,0), ijk_size = None,
             ijk_step = (1,1,1), progress = None, from_cache_only = False):
    '''
    Supported API.
    Return a numpy array for a box shaped subregion of the data with specified
    index origin and size.  Every Nth point can be take along an axis by
    specifying ijk_step.  If step size is greater than 1 then the returned
    array will be smaller than the requested size.  The requested ijk_size
    refers to the region size of the full-resolution array (counting every
    grid point).  The array can be read from a file or be a cached copy in
    memory.  The array should not be modified.
    '''
    if ijk_size is None:
      ijk_size = self.size

    m = self.cached_data(ijk_origin, ijk_size, ijk_step)
    if m is None and not from_cache_only:
      try:
        m = self.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
      except IOError as e:
        import errno
        if e.errno == errno.ENOENT:
          # File not found
          msg = ('\nYou deleted or moved a volume file that is still open in ChimeraX.\n\n%s\n\n'
                 'To allow fast initial display of volume data ChimeraX does not read '
                 'all data from the file when it is first opened, and will later '
                 'read more data when needed. '
                 'ChimeraX got an error trying to read the above file.') % e.filename
          from chimerax.core.errors import UserError
          raise UserError(msg)
        raise

      self.cache_data(m, ijk_origin, ijk_size, ijk_step)

    return m
    
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):
    '''
    Supported API.
    Must overide this function in derived class to return a 3 dimensional
    NumPy matrix.  The returned matrix has size ijk_size and
    element ijk is accessed as m[k,j,i].  It is an error if the requested
    submatrix does not lie completely within the full data matrix.  It is
    also an error for the size to be <= 0 in any dimension.  These invalid
    inputs might throw an exception or might return garbage.  It is the
    callers responsibility to make sure the arguments are valid.
    '''
    if hasattr(self, 'read_xy_plane'):
      m = self._read_full_planes(ijk_origin, ijk_size, ijk_step, progress)
      return m

    raise NotImplementedError('Grid %s has no read_matrix() routine' % self.name)
    
  # ---------------------------------------------------------------------------
  #
  def _read_full_planes(self, ijk_origin, ijk_size, ijk_step, progress):
    '''
    Formats such as TIFF and DICOM image stacks require reading full xy planes.
    If subsampled planes or a subregion are requested, cache the full planes.
    An example where this makes a huge difference is in showing a single xz plane
    where the full data will be read.  If only the plane is cached then moving
    to show another xz plane requires another full read of the data.  This can
    be extremely slow.  With this code the full data is cached.
    '''
    i1, j1, k1 = ijk_origin
    i2, j2, k2 = [i+s-1 for i,s in zip(ijk_origin, ijk_size)]
    istep, jstep, kstep = ijk_step
    xs,ys = self.size[:2]
    f_ijk_size = (xs,ys,1)
    f_ijk_step = (1,1,1)
    planes = []
    for k in range(k1, k2+1, kstep):
      if progress:
        progress.plane((k-k1)//kstep)
      f_ijk_origin = (0,0,k)
      m = self.cached_data(f_ijk_origin, f_ijk_size, f_ijk_step)
      if m is None:
        m2d= self.read_xy_plane(k)
        m = m2d.reshape((1,ys,xs))
        self.cache_data(m, f_ijk_origin, f_ijk_size, f_ijk_step)
      planes.append(m)

    if len(planes) == 1:
      m = planes[0]
      a = m[:, j1:j2+1:jstep, i1:i2+1:istep]
    else:
      mshape = (len(planes), (j2-j1+jstep)//jstep, (i2-i1+istep)//istep)
      from numpy import empty
      a = empty(mshape, self.value_type)
      for k,m in enumerate(planes):
        a[k,:,:] = m[0, j1:j2+1:jstep, i1:i2+1:istep]
      
    return a
  
  # ---------------------------------------------------------------------------
  # Convenience routine.
  #
  def matrix_slice(self, matrix, ijk_origin, ijk_size, ijk_step):

    i1, j1, k1 = ijk_origin
    i2, j2, k2 = [i+s for i,s in zip(ijk_origin, ijk_size)]
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
    if m is not None:
      return m

    if hasattr(self, 'read_xy_plane'):
      return None	# Don't look for subregion when only full planes are cached
    
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

  from chimerax.geometry import Place
  tf = Place(((d0*ax[0], d1*ay[0], d2*az[0], ox),
              (d0*ax[1], d1*ay[1], d2*az[1], oy),
              (d0*ax[2], d1*ay[2], d2*az[2], oz)))
  tf_inv = tf.inverse()
  
  return tf, tf_inv

# -----------------------------------------------------------------------------
# Apply scaling and skewing transformations.
#
def scale_and_skew(ijk, step, cell_angles):

  from chimerax.geometry import place
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
class GridSubregion(GridData):

  def __init__(self, grid_data, ijk_min, ijk_max, ijk_step = (1,1,1)):

    d = grid_data
    self.full_data = d

    ijk_min = [((a+s-1)//s)*s for a,s in zip(ijk_min, ijk_step)]
    self.ijk_offset = ijk_min
    self.ijk_step = ijk_step

    size = [max(0,(b-a+s)//s) for a,b,s in zip(ijk_min, ijk_max, ijk_step)]
    origin = d.ijk_to_xyz(ijk_min)
    step = [ijk_step[a]*d.step[a] for a in range(3)]

    settings = d.settings(size=size, origin=origin, step=step, name=d.name+' subregion')
    GridData.__init__(self, **settings)

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

    origin = [i*s+o for i,s,o in zip(ijk_origin, self.ijk_step, self.ijk_offset)]
    size = [a*b for a,b in zip(ijk_size, self.ijk_step)]
    step = [a*b for a,b in zip(ijk_step, self.ijk_step)]
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
