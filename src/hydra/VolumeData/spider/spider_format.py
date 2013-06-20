# -----------------------------------------------------------------------------
# Read SPIDER volume data into a Python array.
# The file byte order is detected.
#

# -----------------------------------------------------------------------------
#
class SPIDER_Data:

  def __init__(self, path):

    self.path = path

    file = open(path, 'rb')

    self.swap_bytes = self.need_byte_swap(file)

    file.seek(0)
    v = self.read_header_values(file)

    file.seek(0,2)                              # go to end of file
    file_size = file.tell()
    file.close()

    self.check_header_values(v, file_size)
    
    self.data_offset = int(v['labbyt'])
    self.data_size = (int(v['nsam']), int(v['nrow']), int(v['nslice']))

    # Could set step from v['scale'], but documentation does not define the
    # scale value and does not say whether it is always set.  It says it is
    # not used by SPIDER.
    self.data_step = (1,1,1)

    # Could set offset using v['xoff'], v['yoff'], v['zoff'], but documentation
    # does not say whether it is always set.  It says it is not used by SPIDER.
    self.data_origin = (0,0,0)
    
  # ---------------------------------------------------------------------------
  # Infer file byte order from format value.
  #
  def need_byte_swap(self, file):
    
    file.seek(16, 0)
    format = file.read(4)
    import struct
    volfmt = struct.pack('f', 3)
    swapped_volfmt = volfmt[3] + volfmt[2] + volfmt[1] + volfmt[0]
    if format != volfmt and format != swapped_volfmt:
      raise SyntaxError, 'File is not SPIDER volume data.\nBytes 16-19 do not hold float 3 or byte swapped float 3.'
    swap_bytes = (format == swapped_volfmt)
    return swap_bytes
    
  # ---------------------------------------------------------------------------
  # Format derived from spider_format.html documentation.
  #
  def read_header_values(self, file):

    n = 24
    header = file.read(n * 4)
    from numpy import fromstring, float32
    values = fromstring(header, float32)
    if self.swap_bytes:
      values = values.byteswap()

    names = ('nslice', 'nrow', 'irec', 'nhistrec', 'iform', 'imami', 'fmax',
             'fmin', 'av', 'sig', 'ihist', 'nsam', 'labrec', 'iangle', 'phi',
             'theta', 'gamma', 'xoff', 'yoff', 'zoff', 'scale', 'labbyt',
             'lenbyt', 'istack',)

    v = {}
    for k in range(n):
      v[names[k]] = values[k]
    
    return v

  # ---------------------------------------------------------------------------
  #
  def check_header_values(self, v, file_size):

    if v['nslice'] <= 0 or v['nrow'] <= 0 or v['nsam'] <= 0:
      raise SyntaxError, ('Bad SPIDER grid size (%.0f,%.0f,%.0f)'
                          % (v['nslice'],v['nrow'],v['nsam']))

    value_size = 4
    expected_size = v['labbyt'] + value_size*v['nslice']*v['nrow']*v['nsam'] 
    if expected_size > file_size:
      raise SyntaxError, ('File size %d too small for grid size (%.0f,%.0f,%.0f) and data offset %.0f'
                          % (file_size,
                             v['nslice'],v['nrow'],v['nsam'],v['labbyt']))
    
    if v['iform'] != 3:
      raise SyntaxError, ('SPIDER data format %f != 3 (volume data)' % v['iform'])
    if v['istack'] != 0:
      raise SyntaxError, "SPIDER image stacks not supported."

  # ---------------------------------------------------------------------------
  # Reads a submatrix from a potentially very large file.
  # Returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from numpy import float32
    from VolumeData.readarray import read_array
    matrix = read_array(self.path, self.data_offset,
                        ijk_origin, ijk_size, ijk_step,
                        self.data_size, float32, self.swap_bytes,
                        progress)
    return matrix
