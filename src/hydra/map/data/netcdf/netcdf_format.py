# -----------------------------------------------------------------------------
# Read NetCDF 3 dimensional array data.
#
# Use variables or global attributes xyz_origin and xyz_step, each triples
# of floats, to define embedding of 3D data matrices in xyz coordinate space.
#
# 3D array variables can have any name.  Every 3D array is considered.
# Each becomes a separate component.  If they don't all have the same size
# then one size is chosen arbitrarily and only 3D arrays with that size are
# used.
#
# A color can be associated with each 3D array saved as an rgba value
# as an attribute called rgba or with a variable with name matching the
# array name with '_color' appended.
#
# NetCDF doesn't have unsigned integral types.
# To indicate a 3D integer array is supposed to be unsigned save the
# NumPy typecode in an attribute called numpy_typecode or
# in a variable named the same as the 3D array with "_typecode" appended.
#
class NetCDF_Data:

  def __init__(self, path):

    from Scientific.IO import NetCDF
    f = NetCDF.NetCDFFile(path, 'r')

    self.path = path
    self.xyz_origin = self.read_xyz_origin(f)
    self.xyz_step = self.read_xyz_step(f)
    self.cell_angles = self.read_cell_angles(f)
    self.rotation = self.read_rotation(f)
    self.component_name = self.read_component_name(f)
    self.grid_size, self.arrays = self.read_components(f, path)
    
    f.close()
    
  # ---------------------------------------------------------------------------
  #
  def read_xyz_origin(self, f):

    return self.read_vector(f, 'xyz_origin', (0,0,0))
    
  # ---------------------------------------------------------------------------
  #
  def read_xyz_step(self, f):

    return self.read_vector(f, 'xyz_step', (1,1,1))
    
  # ---------------------------------------------------------------------------
  #
  def read_cell_angles(self, f):

    return self.read_vector(f, 'cell_angles', (90,90,90))
    
  # ---------------------------------------------------------------------------
  #
  def read_rotation(self, f):

    axis = self.read_vector(f, 'rotation_axis', None)
    angle = self.read_float(f, 'rotation_angle', None)
    if axis is None or angle is None:
      return ((1,0,0),(0,1,0),(0,0,1))
    import Matrix
    r = Matrix.rotation_from_axis_angle(axis, angle)
    return r

  # ---------------------------------------------------------------------------
  #
  def read_float(self, f, name, default):

    v = default
    if hasattr(f, name):                  # global attribute
      vv = getattr(f, name)
      if len(vv) == 1:
        v = float(vv[0])
    elif f.variables.has_key(name):       # or as a variable
      vv = f.variables[name]
      if vv.shape == (1,):
        v = float(vv[0])
    return v

  # ---------------------------------------------------------------------------
  #
  def read_vector(self, f, name, default):

    v = default
    if hasattr(f, name):                  # global attribute
      vv = tuple(getattr(f, name))
      if len(vv) == 3:
        v = vv
    elif f.variables.has_key(name):       # or as a variable
      vv = f.variables[name]
      if vv.shape == (3,):
        v = tuple(vv[:])
    return v

  # ---------------------------------------------------------------------------
  #
  def read_component_name(self, f):

    comp_name = ''
    if hasattr(f, 'component_name'):                    # global attribute
      comp_name = f.component_name
    elif f.variables.has_key('component_name'):         # or as a variable
      v = f.variables['component_name']
      if len(v.shape) == 1 and v.typecode() == 'c':
        comp_name = v[:].tostring()
    return comp_name
    
  # ---------------------------------------------------------------------------
  #
  def read_components(self, f, path):

    shape = ()
    data_names = []
    for name, var in f.variables.items():
      if len(var.shape) == 3:
        data_names.append(name)
        if not hasattr(var, 'subsample_size'):
          shape = tuple(var.shape)
        
    sort_order = {}
    components = []
    ctable = {}
    for name in data_names:
      var = f.variables[name]
      if var.shape == shape:
        cname = self.read_variable_name(f, name)
        color = self.read_variable_color(f, name)
        if color == None:
          color = self.default_color(len(components))
        typecode = self.read_variable_typecode(f, name)
        c = NetCDF_Array(path, name, cname, typecode, color)
        if hasattr(var, 'component_number'):
          cnum = var.component_number[0]
          ctable[(name, cnum)] = c
          order = str(cnum)
        else:
          order = cname
        sort_order[c] = order
        components.append(c)

    components.sort(lambda c1, c2, so=sort_order: cmp(so[c1], so[c2]))
    
    grid_size = list(shape)
    grid_size.reverse()
    grid_size = tuple(grid_size)

    self.read_subsample_components(f, data_names, ctable)

    return grid_size, components
          
  # ---------------------------------------------------------------------------
  #
  def read_subsample_components(self, f, data_names, ctable):

    for name in data_names:
      var = f.variables[name]
      if (hasattr(var, 'subsamples_of') and
          hasattr(var, 'component_number') and
          hasattr(var, 'subsample_size')):
          k = (var.subsamples_of, var.component_number[0])
          if k in ctable:
              c = ctable[k]
              typecode = self.read_variable_typecode(f, name)
              na = NetCDF_Array(c.path, name, c.descriptive_name, typecode,
                                c.color)
              cell_size = tuple(var.subsample_size)
              if not hasattr(c, 'subsamples'):
                  c.subsamples = {}
              grid_size = list(var.shape)
              grid_size.reverse()
              grid_size = tuple(grid_size)
              c.subsamples[(grid_size, cell_size)] = na
          
  # ---------------------------------------------------------------------------
  #
  def read_variable_name(self, f, name):

    var = f.variables[name]
    if hasattr(var, 'name'):
      return var.name

    return name
  
  # ---------------------------------------------------------------------------
  #
  def read_variable_color(self, f, name):

    color = None
    var = f.variables[name]
    if hasattr(var, 'rgba'):                    # variable attribute
      color = tuple(var.rgba)
    else:
      cname = name + '_color'                   # or separate variable
      if f.variables.has_key(cname):
        cv = f.variables[cname]
        if cv.shape == (4,):
          color = tuple(cv[:])

    return color
  
  # ---------------------------------------------------------------------------
  #
  def default_color(self, n):

    colors = (
      (1,1,1,1),           # white
      (1,0,0,1),           # red
      (0,1,0,1),           # green
      (0,0,1,1),           # blue
      (1,1,0,1),           # yellow
      (0,1,1,1),           # cyan
      )
    color = colors[n % len(colors)]

    return color

  # ---------------------------------------------------------------------------
  #
  def read_variable_typecode(self, f, name):

    var = f.variables[name]
    if hasattr(var, 'numeric_python_typecode'):         # variable attribute
      typecode = var.numeric_python_typecode
      if typecode in numeric_typecode_to_numpy:
        typecode = numeric_typecode_to_numpy[typecode]
    elif hasattr(var, 'numpy_typecode'):
      typecode = var.numpy_typecode
    else:
      tname = name + '_typecode'
      if f.variables.has_key(tname):
        typecode = f.variables[tname].getValue().tostring()
      else:
        typecode = var.typecode()
    return typecode
    
# -----------------------------------------------------------------------------
#
class NetCDF_Array:

  def __init__(self, path, variable_name, descriptive_name, typecode, color):

    self.path = path
    self.variable_name = variable_name
    self.descriptive_name = descriptive_name
    from numpy import dtype
    self.dtype = dtype(typecode)
    self.color = color
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from Scientific.IO import NetCDF
    f = NetCDF.NetCDFFile(self.path, 'r')
    if progress:
      progress.close_on_cancel(f)
    v = f.variables[self.variable_name]
    i1, j1, k1 = ijk_origin
    i2, j2, k2 = map(lambda i, s: i+s, ijk_origin, ijk_size)
    istep, jstep, kstep = ijk_step
    from ..readarray import allocate_array
    m = allocate_array(ijk_size, self.dtype, ijk_step, progress)
    for k in range(k1,k2,kstep):
      # Cast of istep and jstep to int works around ScientificPython/netcdf bug
      # where numpy.int64 step gives wrong array size (as if step = 1 when
      # step = 2), Chimera bug 8965.
      m[(k-k1)/kstep,:,:] = v[k, j1:j2:int(jstep), i1:i2:int(istep)].view(self.dtype)
      if progress:
        progress.plane((k-k1)/kstep)
    f.close()
    return m

# -----------------------------------------------------------------------------
#
def write_grid_as_netcdf(grid_data, outpath, options = {}, progress = None):

  from Scientific.IO import NetCDF
  f = NetCDF.NetCDFFile(outpath, 'w')
  if progress:
    progress.close_on_cancel(f)

  # createDimension() cannot handle long integer size values
  xsize, ysize, zsize = map(int, grid_data.size)
  f.createDimension('x', xsize)
  f.createDimension('y', ysize)
  f.createDimension('z', zsize)

  f.xyz_origin = grid_data.origin
  f.xyz_step = grid_data.step
  if grid_data.cell_angles != (90,90,90):
    f.cell_angles = grid_data.cell_angles
  if grid_data.rotation != ((1,0,0),(0,1,0),(0,0,1)):
    import Matrix
    axis, angle = Matrix.rotation_axis_angle(grid_data.rotation)
    f.rotation_axis = axis
    f.rotation_angle = angle

  name = 'data'
  typecode = grid_data.value_type.char
  v = f.createVariable(name, typecode, ('z','y','x'))
  v.rgba = grid_data.rgba
  v.component_number = 1
  save_unsigned_typecode(v, typecode)
  sarrays = subsample_arrays(grid_data, name, f)
  for k in range(zsize):
    if progress:
      progress.plane(k)
    values = grid_data.matrix((0,0,k), (xsize,ysize,1))
    v[k,:,:] = values.view(v.typecode())[0,:,:]
    for cell_size, ssv in sarrays:
      kstep = cell_size[2]
      if k % kstep == 0:
        ssd = grid_data.available_subsamplings[cell_size]
        xs,ys,zs = ssd.size
        ssk = k/kstep
        if ssk < zs:
          values = ssd.matrix((0,0,ssk), (xs,ys,1))
          ssv[ssk,:,:] = values.view(ssv.typecode())[0,:,:]

  # Subsample arrays may have an extra plane.
  for cell_size, ssv in sarrays:
    ssd = grid_data.available_subsamplings[cell_size]
    xs,ys,zs = ssd.size
    for ssk in range(zsize/cell_size[2], zs):
      values = ssd.matrix((0,0,ssk), (xs,ys,1))
      ssv[ssk,:,:] = values.view(ssv.typecode())[0,:,:]
    
  f.close()

# -----------------------------------------------------------------------------
# Created variables for subsample arrays.
#
def subsample_arrays(grid_data, dname, f):

  sarrays = []

  from . import Subsampled_Grid
  if not isinstance(grid_data, Subsampled_Grid):
    return sarrays

  cslist = grid_data.available_subsamplings.keys()
  cslist.remove((1,1,1))
  cslist.sort()
  for cell_size in cslist:
    ss_grid_data = grid_data.available_subsamplings[cell_size]
    xsize, ysize, zsize = map(int, ss_grid_data.size)
    suffix = '_%d%d%d' % cell_size
    xname, yname, zname = 'x'+suffix, 'y'+suffix, 'z'+suffix
    f.createDimension(xname, xsize)
    f.createDimension(yname, ysize)
    f.createDimension(zname, zsize)
    name = '%s_subsamples_%d_%d_%d' % ((dname,) + cell_size)
    typecode = grid_data.value_type.char
    v = f.createVariable(name, typecode, (zname,yname,xname))
    v.subsample_size = cell_size
    v.subsamples_of = dname
    v.component_number = 1
    save_unsigned_typecode(v, typecode)
    sarrays.append((cell_size, v))

  return sarrays
  
# -----------------------------------------------------------------------------
# NetCDF does not have unsigned integer types.  A NetCDF variable created
# with an unsigned typecode (e.g. B, H, L for uint8, uint16, uint32) actually
# reports having a signed typecode (e.g. b, h, l).
#
# Remember the unsigned type in another variable.
#
def save_unsigned_typecode(var, value_type):

  if var.typecode() != value_type:
    var.numpy_typecode = value_type

# -----------------------------------------------------------------------------
#
def legalize_variable_name(name):

  n = ''
  for c in name:
    if (c.isalnum() or c in '-_'):
      n = n + c
  return n

# -----------------------------------------------------------------------------
# Conversions from Numeric to NumPy type codes.
#
numeric_typecode_to_numpy = {'b':'B', '1':'b', 's':'h', 'w':'H', 'u':'I'}
