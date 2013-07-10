# -----------------------------------------------------------------------------
# Read TOM Toolbox EM density map file (http://www.biochem.mpg.de/tom/)
# electron microscope data.
#
# Byte swapping will be done if needed.
#

# -----------------------------------------------------------------------------
#
class EM_Data:

  def __init__(self, path):

    self.path = path

    import os.path
    self.name = os.path.basename(path)
    
    file = open(path, 'rb')

    file.seek(0,2)                              # go to end of file
    file_size = file.tell()
    file.seek(0,0)                              # go to beginning of file

    # Determine byte order from machine code
    #  OS-9         0
    #  VAX          1
    #  Convex       2
    #  SGI          3
    #  Sun          4 (not supported)
    #  Mac          5
    #  PC           6
    self.swap_bytes = False
    from numpy import int8, little_endian
    machine_code = self.read_values(file, int8, 1)
    file_little_endian = machine_code in (1, 6)
    self.swap_bytes = ((file_little_endian and not little_endian) or
                       (not file_little_endian and little_endian))
    file.seek(0,0)

    v = self.read_header_values(file)
    self.check_header_values(v, file_size)
    
    self.data_offset = file.tell()
    file.close()

    self.data_size = (v['xsize'], v['ysize'], v['zsize'])
    dstep = v['pixelsize']
    if dstep == 0:
      dstep = 1.0
    self.data_step = (dstep, dstep, dstep)
    self.data_origin = (0., 0., 0.)
    
  # ---------------------------------------------------------------------------
  # Format derived from C header file mrc.h.
  #
  def read_header_values(self, file):

    from numpy import int8, int32
    i8 = int8
    i32 = int32
    
    v = {}
    v['machine code']= self.read_values(file, i8, 1)
    v['os 9 version']= self.read_values(file, i8, 1)
    v['abandoned header']= self.read_values(file, i8, 1)
    v['data type code']= self.read_values(file, i8, 1)
    v['xsize'], v['ysize'], v['zsize'] = self.read_values(file, i32, 3)
    v['comment'] = file.read(80)
    v['user param'] = self.read_values(file, i32, 40)
    v['pixelsize'] = v['user param'][6] / 1000.0        # nm
    v['user data'] = file.read(256)
    
    return v

  # ---------------------------------------------------------------------------
  #
  def check_header_values(self, v, file_size):

    mc = v['machine code']
    if mc < 0 or mc > 6:
      raise SyntaxError, ('Bad EM machine code %d at byte 0, must be 0 - 6.'
                          % mc)

    dc = v['data type code']
    if not dc in (1,2,4,5,8,9):
      raise SyntaxError, ('Bad EM data type code %d' % dc +
                          ', must be 1, 2, 4, 5, 8, or 9')

    from numpy import uint8, int16, int32, float32, float64
    types = { 1: uint8,
              2: int16,
              4: int32,
              5: float32,
              9: float64 }
    if types.has_key(dc):
      self.element_type = types[dc]
    else:
      raise SyntaxError, 'Complex EM data value type not supported'

    if float(v['xsize']) * float(v['ysize']) * float(v['zsize']) > file_size:
      raise SyntaxError, ('File size %d too small for grid size (%d,%d,%d)'
                          % (file_size, v['xsize'],v['ysize'],v['zsize']))

  # ---------------------------------------------------------------------------
  #
  def read_values(self, file, etype, count):

    from numpy import array
    esize = array((), etype).itemsize
    string = file.read(esize * count)
    values = self.read_values_from_string(string, etype, count)
    return values

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
  # Returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from VolumeData.readarray import read_array
    matrix = read_array(self.path, self.data_offset,
                        ijk_origin, ijk_size, ijk_step,
                        self.data_size, self.element_type, self.swap_bytes,
                        progress)
    return matrix
