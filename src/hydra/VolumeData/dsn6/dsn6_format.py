# -----------------------------------------------------------------------------
# Read a DSN6 or BRIX electron density map into a Python array.
#
# The DSN6 files are fortran integer*2 in big-endian byte order.
# Even the single byte data values are written this way.
#

# -----------------------------------------------------------------------------
#
class dsn6_map:

  def __init__(self, path):

    self.path = path

    file = open(path, 'rb')
    format_id = file.read(3)
    file.seek(0)

    if format_id == ':-)':
      read_header = self.read_brix_header
      self.format = 'brix'
    else:
      read_header = self.read_dsn6_header
      self.format = 'dsn6'

    (self.origin,               # origin of map in grid units
     self.extent,               # size of this map
     self.grid,                 # unit cell size
     self.cell,                 # unit cell parameters
     self.prod,                 # 255.0 / (rhomax - rhomin)
     self.plus,                 # -rhomin * prod
     ) = read_header(file)
    self.scale_values = True

    self.data_offset = file.tell()

    xsize, ysize, zsize = self.extent
    if xsize < 0 or ysize < 0 or zsize < 0:
      raise SyntaxError('Negative grid size in header %s' % str(self.extent))
    
    file.seek(0,2)                      # go to end of file
    file_size = file.tell()
    if float(xsize) * float(ysize) * float(zsize) > file_size:
      raise SyntaxError(('File size %d is too small' % file_size +
                         ' for grid size listed in header %s'
                         % str(self.extent)))
        
    file.close()

  # ---------------------------------------------------------------------------
  #
  def read_brix_header(self, file):

    header = file.read(512)

    if header[:3] != ":-)":
      raise SyntaxError('File does not start with :-)')

    fields = (
      ":-)",
      " origin", (3, int, 5),
      " extent", (3, int, 5),
      " grid", (3, int, 5),
      " cell ", (6, float, 10),
      " prod", (1, float, 12),
      " plus", (1, int, 8), 
      " sigma ", (1, float, 12),
      )

    error_msg = 'Incorrect ASCII header format at byte %d.\n":-) origin", 3i5," extent", 3i5, " grid", 3i5, " cell ", 6f10.3, " prod", f12.5, " plus", i8, " sigma ", f12.5'

    offset = 0
    values = []
    for f in fields:
      if isinstance(f, basestring):
        size = len(f)
        if header[offset:offset+size].lower() != f:
          raise SyntaxError(error_msg % offset)
        offset += size
      else:
        count, to_number, size = f
        field_value = []
        for c in range(count):
          try:
            v = to_number(header[offset:offset+size])
          except ValueError:
            raise SyntaxError(error_msg % offset)
          field_value.append(v)
          offset += size
        if len(field_value) == 1:
          field_value = field_value[0]
        else:
          from numpy import array
          field_value = array(field_value)
        values.append(field_value)

    return values[:6]   # omit sigma value

  # ---------------------------------------------------------------------------
  #
  def read_dsn6_header(self, file):

    origin = self.read_integers(file, 3)
    extent = self.read_integers(file, 3)
    grid = self.read_integers(file, 3)
    i1cell = self.read_integers(file, 6)
    i2prod = self.read_integers(file, 1)[0]
    plus = self.read_integers(file, 1)[0]
    i1 = self.read_integers(file, 1)[0]
    i2 = self.read_integers(file, 1)[0]

    if i1 == 0:
      raise SyntaxError('Unit cell normalization parameter is 0.')

    cell = i1cell / float(i1)

    if i2 == 0:
      raise SyntaxError('Scale factor normalization parameter is 0.')

    prod = float(i2prod) / i2

    self.read_integers(file, 256 - 19)                  # zeros

    return origin, extent, grid, cell, prod, plus

  # ---------------------------------------------------------------------------
  # Assume big-endian byte order in file.
  #
  def read_integers(self, file, count):

    value_string = file.read(2 * count)
    if len(value_string) < 2 * count:
      raise SyntaxError('Unexpected end of file reading %d bytes' % (2*count))

    from numpy import fromstring, int16, little_endian
    values = fromstring(value_string, int16)
    
    if little_endian:
      values = values.byteswap()
      
    return values
  
  # ---------------------------------------------------------------------------
  #
  def matrix(self, progress):

    file = open(self.path, 'rb')
    file.seek(self.data_offset)
    if progress:
      progress.close_on_cancel(file)
    data = self.read_data(file, progress)       # 3D byte array
    file.close()

    return data
  
  # ---------------------------------------------------------------------------
  # Data is in 8 by 8 by 8 blocks of bytes.
  #
  def read_data(self, file, progress):

    block_size = 8
    bsize3 = block_size*block_size*block_size
    xsize, ysize, zsize = self.extent
    from numpy import zeros, float32, fromstring, int16, uint8
    data = zeros((zsize, ysize, xsize), float32)
    xblocks, yblocks, zblocks = ((self.extent - 1) // block_size) + 1
    for zblock in range(zblocks):
      z = block_size * zblock
      zbsize = min(z + block_size, zsize) - z
      if progress:
        progress.fraction(float(zblock)/zblocks)
      for yblock in range(yblocks):
        y = block_size * yblock
        ybsize = min(y + block_size, ysize) - y
        for xblock in range(xblocks):
          x = block_size * xblock
          xbsize = min(x + block_size, xsize) - x
          bytes = file.read(bsize3)
          if self.format == 'dsn6':
            # Swap each pair of bytes for DSN6 format.
            btemp = fromstring(bytes, int16)
            swapped_bytes = btemp.byteswap().tostring()
            bdata = fromstring(swapped_bytes, uint8)
          else:
            bdata = fromstring(bytes, uint8)
          bdata3d = bdata.reshape(block_size,block_size,block_size)
          data[z:z+zbsize, y:y+ybsize, x:x+xbsize] = \
                           bdata3d[0:zbsize, 0:ybsize, 0:xbsize]

    if self.scale_values and self.prod != 0:
      data -= self.plus
      data /= self.prod

    if progress:
      progress.done()

    return data
