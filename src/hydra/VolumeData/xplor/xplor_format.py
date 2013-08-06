# -----------------------------------------------------------------------------
# Read a CNS or XPLOR ascii density map file.
#

# -----------------------------------------------------------------------------
#
class FileFormatError(Exception):
  pass

# -----------------------------------------------------------------------------
#
class XPLOR_Density_Map:

  def __init__(self, path):

    self.path = path
    #
    # Open file in binary mode 'rb'.  Opening in mode 'r' in Python 2.4.2
    # on Windows with '\n' line endings gives incorrect f.tell() values,
    #
    f = open(path, 'rb')

    f.readline()                          # First line is blank
    ntitle_line = f.readline()            # integer number of comment lines
    try:
      ntitle = int(ntitle_line.split()[0])
    except:
      raise SyntaxError, ('Invalid XPLOR comment line count on line 2: %s'
                          % ntitle_line[:80])

    comments = []
    for t in range(ntitle):
      comments.append(f.readline())
    self.comments = comments

    extent_line = f.readline()
    from VolumeData.readarray import split_fields
    extent = split_fields(extent_line, 8, 9)
    try:
      na, amin, amax, nb, bmin, bmax, nc, cmin, cmax = map(int, extent)
    except ValueError:
      raise SyntaxError, 'Invalid XPLOR grid size line: %s' % extent_line[:80]
      
    self.na, self.amin, self.amax = na, amin, amax
    self.nb, self.bmin, self.bmax = nb, bmin, bmax
    self.nc, self.cmin, self.cmax = nc, cmin, cmax
        
    cell_line = f.readline()
    cell = split_fields(cell_line, 12, 6)
    try:
      cell_params = map(float, cell)
    except ValueError:
      raise SyntaxError, ('Invalid XPLOR cell parameters line: %s'
                          % cell_line[:80])
    self.cell_size = cell_params[0:3]
    self.cell_angles = cell_params[3:6]

    axis_order_line = f.readline()        # Must be ZYX

    asize, bsize, csize = (amax - amin + 1, bmax - bmin + 1, cmax - cmin + 1)
    if asize <= 0 or bsize <= 0 or csize <=0:
      raise SyntaxError, ('Bad XPLOR grid size (%d,%d,%d)'
                          % (asize, bsize, csize))
    self.grid_size = (asize, bsize, csize)

    self.data_offset = f.tell()
    f.seek(0,2)                         # End of file
    self.file_size = f.tell()
    
    f.close()

  # ---------------------------------------------------------------------------
  #
  def matrix(self, progress):

    f = open(self.path, 'rb')
    f.seek(self.data_offset)

    if progress:
      progress.text_file_size(self.file_size)
      progress.close_on_cancel(f)
    
    asize, bsize, csize = self.grid_size
    from numpy import zeros, float32, array, reshape
    data = zeros((csize, bsize, asize), float32)
    from VolumeData.readarray import read_float_lines
    for c in range(csize):
      if progress:
        progress.plane(c)
      f.readline()                        # c section number
      read_float_lines(f, data[c,:,:], line_format = (12,6))

    f.readline()                          # footer - int value must be -9999
    f.readline()                          # density average and standard dev

    f.close()

    return data
