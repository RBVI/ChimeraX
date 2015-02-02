# -----------------------------------------------------------------------------
# Read UHBD binary grid file.
# Byte swapping will be done if needed.
#

# -----------------------------------------------------------------------------
#
class UHBD_Data:

  def __init__(self, path):

    self.path = path
    import os.path
    self.name = os.path.basename(path)
    
    file = open(path, 'rb')

    file.seek(0,2)                              # go to end of file
    file_size = file.tell()
    file.seek(0,0)                              # go to beginning of file

    self.swap_bytes = self.determine_byte_order(file)

    v = self.read_header_values(file)
    self.check_header_values(v, file_size)
    
    self.data_offset = file.tell()

    self.title = v['title']
    self.data_scale = v['scale']
    self.data_size = (v['im'], v['jm'], v['km'])
    h = v['h']
    self.data_step = (h, h, h)
    #
    # Origin is offset by one voxel because first grid position is (1,1,1)
    # due to fortran array indexing and (ox+i*h,oy+j*h,oz+k*h) is position
    # for grid value (i,j,k).
    #
    self.data_origin = (v['ox'] + h, v['oy'] + h, v['oz'] + h)

    file.close()
    
  # ---------------------------------------------------------------------------
  # Infer file byte order using interger value 1 saved in header.
  #
  def determine_byte_order(self, file):

    file.seek(96,0)
    self.swap_bytes = 0
    from numpy import int32
    one = self.read_values(file, int32, 1)
    if one != 1 and one != 16777216:
      file.close()
      raise SyntaxError, ('UHBD file does not contain integer value 1 at bytes 96-99\n using either little or big endian byte order. Found %d' % one)
    swap_bytes = not (one == 1)
    file.seek(0,0)
    return swap_bytes
    
  # ---------------------------------------------------------------------------
  # Format derived from uhbd_asc2bin.f source code.
  #
  def read_header_values(self, file):

    from numpy import int32, float32
    i32 = int32
    f32 = float32

    v = {}
    v['reclenbegin'] = self.read_values(file, i32, 1)
    v['title'] = file.read(72)
    v['scale'] = self.read_values(file, f32, 1) # scale factor for data values
    v['dum2'] = self.read_values(file, f32, 1)
    v['grdflg'] = self.read_values(file, i32, 1)     # indicates type of grid
    v['idum2'], v['km1'], v['one'], v['km2'] = self.read_values(file, i32, 4)
    v['im'], v['jm'], v['km'] = self.read_values(file, i32, 3) # grid size
    v['h'] = self.read_values(file, f32, 1)          # voxel size
    v['ox'], v['oy'], v['oz'] = self.read_values(file, f32, 3) # grid origin
    v['dum3'], v['dum4'], v['dum5'], v['dum6'], v['dum7'], v['dum8'] = \
               self.read_values(file, f32, 6)
    v['idum3'], v['idum4'] = self.read_values(file, i32, 2)
    v['reclenend'] = self.read_values(file, i32, 1)
    
    return v

  # ---------------------------------------------------------------------------
  #
  def check_header_values(self, v, file_size):

    if v['im'] <= 0 or v['jm'] <= 0 or v['km'] <= 0:
      raise SyntaxError, ('Bad UHBD grid size (%d,%d,%d)'
                          % (v['im'],v['jm'],v['km']))

    if 4 * float(v['im']) * float(v['jm']) * float(v['km']) > file_size:
      raise SyntaxError, ('File size %d too small for grid size (%d,%d,%d)'
                          % (file_size, v['im'],v['jm'],v['km']))

    if v['h'] <= 0:
      raise SyntaxError, ('Bad UHBD voxel size %g <= 0' % v['h'])

    if v['scale'] <= 0:
      raise SyntaxError, ('Bad UHBD data scale factor %g <= 0' % v['scale'])


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
  # Binary file is written in unformatted Fortran style with record length
  # proceeding and following each record.  Matrix is written one z plane at
  # a time with plane number and size (k, im, jm) in a record preceding each
  # plane.
  #
  def matrix(self, progress):

    xsize, ysize, zsize = self.data_size
    plane_bytes = 4 * xsize * ysize

    from numpy import zeros, float32, fromstring, reshape
    matrix = zeros((zsize, ysize*xsize), float32)

    file = open(self.path, 'rb')
    file.seek(self.data_offset, 0)
    if progress:
      progress.close_on_cancel(file)

    for k in range(zsize):
      if progress:
        progress.plane(k)
      file.seek(24,1)  # skip rec-len-beg, k, im, jm, rec-len-end, rec-len-beg
      plane = file.read(plane_bytes)
      matrix[k,:] = fromstring(plane, float32)
      file.seek(4,1)   # skip rec-len-end

    file.close()

    if self.swap_bytes:
      matrix.byteswap(True)

    matrix = reshape(matrix, (zsize, ysize, xsize))

    # TODO: Might want to scale by self.data_scale.  The uhbd to delphi
    # conversion program gridbin2grasp.f does not do the scaling.

    return matrix
