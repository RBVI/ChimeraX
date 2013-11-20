# -----------------------------------------------------------------------------
# Read Purdue Image format electron microscope data.
# Byte swapping will be done if needed.
#

# -----------------------------------------------------------------------------
#
class PIF_Data:

  def __init__(self, path):

    self.path = path

    import os.path
    self.name = os.path.basename(path)
    
    file = open(path, 'rb')

    file.seek(0,2)                              # go to end of file
    file_size = file.tell()
    file.seek(0,0)                              # go to beginning of file

    self.swap_bytes = self.is_byte_swap_needed(file)

    v = self.read_header_values(file)
    self.check_header_values(v, file_size, file)
    self.header = v     # For use by dumppif.py
    
    self.data_offset = file.tell()

    file.close()

    (self.ijk_to_crs,
     self.crs_to_ijk,
     self.matrix_size,
     self.data_size,
     self.data_step,
     self.cell_angles,
     self.data_origin,
     self.min_intensity,
     self.max_intensity) = self.header_values(v)
    
  # ---------------------------------------------------------------------------
  # Assume file is positioned at byte 0.
  #
  def is_byte_swap_needed(self, file):

    PIF_MAGIC_NUMBER = 8
    PIF_MAGIC_NUMBER_BYTE_SWAPPED = (8 << 24)

    # Infer file byte order from header magic number.
    self.swap_bytes = False  # Need this set before calling read_int32().
    mnum = self.read_int32(file, 1)
    file.seek(0,0)
    if mnum == PIF_MAGIC_NUMBER:
      swap = False
    elif mnum == PIF_MAGIC_NUMBER_BYTE_SWAPPED:
      swap = True
    else:
      raise SyntaxError, ('First 4 bytes of PIF file %s must be %d, got %d'
                          % (file.name, PIF_MAGIC_NUMBER, mnum))
    return swap
    
  # ---------------------------------------------------------------------------
  # Format derived from C header file mrc.h.
  #
  def read_header_values(self, file):

    v = {}

    from numpy import int32
    i32 = int32

    v['magic_1'], v['magic_2'] = self.read_int32(file, 2)
    v['realScaleFactor'] = file.read(16)
    v['numImages'] = self.read_int32(file, 1)
    v['endianNess'] = self.read_int32(file, 1)
    v['genProgram'] = file.read(32)
    v['htype'] = self.read_int32(file, 1)
    v['nx'], v['ny'], v['nz'] = self.read_int32(file, 3)
    v['mode'] = self.read_int32(file, 1)
    v['even'] = self.read_int32(file, 1)
    v['mrcX'], v['mrcY'], v['mrcZ'] = self.read_int32(file, 3)
    v['int4scaleFstr'] = file.read(16)
    extra = file.read(396)

    try:
      ff = float(v['realScaleFactor'])
    except:
      raise SyntaxError, ('realScaleFactor header value of PIF file %s is not parsable as a float: "%s"' % (file.name, v['realScaleFactor']))
    
    v['imageHeader'] = self.read_image_header(file, ff)
    
    return v
    
  # ---------------------------------------------------------------------------
  # Each image has a file header similar to MRC format headers.
  #
  def read_image_header(self, file, ff):

    v = {}
    v['nx'], v['ny'], v['nz'] = self.read_int32(file, 3)
    v['mode'] = self.read_int32(file, 1)
    v['bkgnd'] = self.read_int32(file, 1)
    v['packRadius'] = self.read_int32(file, 1)
    v['nxstart'], v['nystart'], v['nzstart'] = self.read_int32(file, 3)
    v['mx'], v['my'], v['mz'] = self.read_int32(file, 3)
    v['xlength'], v['ylength'], v['zlength'] = self.read_floats(file, 3, ff)
    v['alpha'], v['beta'], v['gamma'] = self.read_floats(file, 3, ff)
    v['mapc'], v['mapr'], v['maps'] = self.read_int32(file, 3)
    v['min'], v['max'], v['mean'], v['stdDev'] = self.read_floats(file, 4, ff)
    v['ispg'], v['nsymbt'] = self.read_int32(file, 2)
    v['xorigin'], v['yorigin'] = self.read_floats(file, 2, ff)
    v['title/description'] = file.read(80)
    v['timeStamp'] = file.read(32)
    v['microGraphDesignation'] = file.read(16)
    v['scanNumber'] = file.read(8)
    v['aoverb'] = self.read_floats(file, 1, ff)
    v['map_abang'] = self.read_floats(file, 1, ff)
    v['dela'], v['delb'], v['delc'] = self.read_floats(file, 3, ff)
    v['t_matrix'] = self.read_floats(file, 6, ff)
    v['dthe'] = self.read_floats(file, 1, ff)
    v['dphi_90'] = self.read_floats(file, 1, ff)
    v['symmetry'] = self.read_int32(file, 1)
    v['binFactor'] = self.read_int32(file, 1)
    v['a_star'], v['b_star'], v['c_star'] = self.read_floats(file, 3, ff)
    v['alp_star'], v['bet_star'], v['gam_star'] = self.read_floats(file, 3, ff)
    v['pixelSize'] = self.read_floats(file, 1, ff)
    v['pixSizeUnits'] = self.read_int32(file, 1)
    v['res_out'] = self.read_floats(file, 1, ff)
    v['ctfMode'] = self.read_int32(file, 1)
    v['tempFactor'] = self.read_floats(file, 1, ff)
    v['wiener'] = self.read_floats(file, 1, ff)
    v['borderType'] = self.read_int32(file, 1)
    v['borderWidth'] = self.read_int32(file, 1)
    v['r4_min'], v['r4_max'] = self.read_floats(file, 2, ff)
    v['rad_lo'], v['rad_hi'] = self.read_int32(file, 2)
    v['transX'], v['transY'], v['transZ'] = self.read_int32(file, 3)
    v['isTrunc'] = self.read_int32(file, 1)
    v['origX'], v['origY'], v['origZ'] = self.read_int32(file, 3)
    extra = file.read(100)
    
    return v

  # ---------------------------------------------------------------------------
  #
  def check_header_values(self, v, file_size, file):

    vi = v['imageHeader']
    m = vi['mode']

    self.float_scale = None
    try:
      fscale = float(v['realScaleFactor'])
    except:
      fscale = None
      
    from numpy import int16, int32, float32
    if m == 20 or m == 7:
      self.file_element_type = int16
      self.file_element_size = 2
      self.float_scale = fscale
    elif m == 21 or m == 2 or m == 22:
      self.file_element_type = int32
      self.file_element_size == 4
      self.float_scale = fscale
    elif m == 1:
      self.file_element_type = int16
      self.file_element_size = 2
    elif m == 9:
      self.file_element_type = float32
      self.file_element_size = 4
    else:
      raise SyntaxError, ('PIF data value type %d not supported ' % m)

    if self.float_scale != None:
      self.element_type = float32
    else:
      self.element_type = self.file_element_type

    if vi['nx'] <= 0 or vi['ny'] <= 0 or vi['nz'] <= 0:
      raise SyntaxError, ('Bad PIF grid size (%d,%d,%d)'
                          % (vi['nx'],vi['ny'],vi['nz']))

    data_size = int(vi['nx'])*int(vi['ny'])*int(vi['nz'])*self.file_element_size
    header_end = file.tell()
    if header_end + data_size > file_size:
      raise SyntaxError, ('File size %d too small for grid size (%d,%d,%d)'
                          % (file_size, vi['nx'],vi['ny'],vi['nz']))

  # ---------------------------------------------------------------------------
  #
  def header_values(self, v):

    vi = v['imageHeader']
    mapc, mapr, maps = vi['mapc'], vi['mapr'], vi['maps']
    if (1 in (mapc, mapr, maps) and
        2 in (mapc, mapr, maps) and
        3 in (mapc, mapr, maps)):
      crs_to_ijk = (mapc-1,mapr-1,maps-1)
      ijk_to_crs = [None,None,None]
      for a in range(3):
        ijk_to_crs[crs_to_ijk[a]] = a
    else:
      crs_to_ijk = ijk_to_crs = (0, 1, 2)
    self.crs_to_ijk = crs_to_ijk
    self.ijk_to_crs = ijk_to_crs

    crs_size = vi['nx'], vi['ny'], vi['nz']
    matrix_size = crs_size
    data_size = [crs_size[a] for a in ijk_to_crs]
    
    mx, my, mz = vi['mx'], vi['my'], vi['mz']
    xlen, ylen, zlen = vi['xlength'], vi['ylength'], vi['zlength']
    if mx > 0 and my > 0 and mz > 0 and xlen > 0 and ylen > 0 and zlen > 0:
      data_step = (xlen/mx, ylen/my, zlen/mz)
    else:
      data_step = (1.0, 1.0, 1.0)

#  Cell angles are uninitialized in some files.  For EM data unlikely to
#  have anything other than orthogonal cell.
#    cell_angles = (vi['alpha'], vi['beta'], vi['gamma'])
    cell_angles = (90, 90, 90)

    crs_start = vi['nxstart'], vi['nystart'], vi['nzstart']
    ijk_start = [crs_start[a] for a in ijk_to_crs]
    nxs, nys, nzs = ijk_start
    if (nxs >= -mx and nxs < mx and
        nys >= -my and nys < my and
        nzs >= -mz and nzs < mz):
      from VolumeData.griddata import scale_and_skew
      data_origin = scale_and_skew(ijk_start, data_step, cell_angles)
    else:
      data_origin = (0., 0., 0.)
    
    min_intensity = vi['min']
    max_intensity = vi['max']

    return (ijk_to_crs,
            crs_to_ijk,
            matrix_size,
            data_size,
            data_step,
            cell_angles,
            data_origin,
            min_intensity,
            max_intensity)
            
  # ---------------------------------------------------------------------------
  #
  def read_int32(self, file, count):

    from numpy import array, int32
    esize = array((), int32).itemsize
    string = file.read(esize * count)
    values = self.read_values_from_string(string, int32, count)
    return values
            
  # ---------------------------------------------------------------------------
  #
  def read_floats(self, file, count, ff):

    i = self.read_int32(file, count)
    if count == 1:
      return i*ff
    from numpy import array, float32
    f = i * array(ff, float32)
    return f
  
  # ---------------------------------------------------------------------------
  #
  def read_values_from_string(self, string, etype, count):
  
    from numpy import fromstring
    values = fromstring(string, etype)
    if self.swap_bytes:
      values = values.byteswap()
    if count == 1:
      return values[0]
    return values

  # ---------------------------------------------------------------------------
  # Reads a submatrix from a potentially very large file.
  # Returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    # ijk correspond to xyz.  crs refers to fast,medium,slow matrix file axes.
    crs_origin = [ijk_origin[a] for a in self.crs_to_ijk]
    crs_size = [ijk_size[a] for a in self.crs_to_ijk]
    crs_step = [ijk_step[a] for a in self.crs_to_ijk]

    from VolumeData.readarray import read_array
    matrix = read_array(self.path, self.data_offset,
                        crs_origin, crs_size, crs_step, self.matrix_size,
                        self.file_element_type, self.swap_bytes,
                        progress)
    matrix = self.permute_matrix_to_xyz_axis_order(matrix)
    
    if self.element_type != self.file_element_type:
      matrix = matrix.astype(self.element_type)
      
    if self.float_scale != None:
      matrix *= self.float_scale
    
    return matrix

  # ---------------------------------------------------------------------------
  #
  def permute_matrix_to_xyz_axis_order(self, matrix):
    
    if self.ijk_to_crs == (0,1,2):
      return matrix

    kji_to_src = [2-self.ijk_to_crs[2-a] for a in (0,1,2)]
    m = matrix.transpose(kji_to_src)

    return m
