# -----------------------------------------------------------------------------
# Read DelPhi or GRASP unformatted phi file.  This was derived from the
# Chimera DelphiViewer extension file reading code.
#

# -----------------------------------------------------------------------------
#
class DelPhi_Data:

  def __init__(self, path):

    self.path = path

    file = open(path, 'rb')

    file.seek(0,2)                              # goto end of file
    self.file_size = file.tell()
    file.seek(0,0)                              # goto beginning of file

    if self.file_size == 0:
      raise SyntaxError, 'Empty file'

    swap = self.need_to_swap_bytes(file)
    uplbl = self.read_record(file, swap)
    morelabels = self.read_record(file, swap)
    self.data_offset = file.tell()
    self.skip_record(file, swap)
    botlbl = self.read_record(file, swap)
    params = self.read_record(file, swap)
    file.close()

    from numpy import float32, int32, float64
    if len(params) == 16:	# GRASP Phi file
      size = 65
      self.value_type = float32
      sc = params
    elif len(params) == 20:	# DelPhi Phi file
      size = string_values(params[16:20], int32, swap)[0]
      self.value_type = float32
      sc = params[:16]
    elif len(params) == 36:	# 2008 Mac DelPhi Phi file
      size = string_values(params[32:36], int32, swap)[0]
      self.value_type = float64
      sc = params[:32]
    else:
      raise SyntaxError, ('Parameter record size %d must be 16 or 20 or 36'
                          % len(params))
    pval = string_values(sc, self.value_type, swap)
    scale = pval[0]
    xyz_center = pval[1:4]

    step = 1.0/scale
    half_size = step * ((size - 1) / 2)
    xyz_origin = map(lambda c, hs=half_size: c - hs, xyz_center)

    self.swap = swap
    self.size = (size, size, size)
    self.xyz_step = (step, step, step)
    self.xyz_origin = xyz_origin
    
  # ---------------------------------------------------------------------------
  #
  def need_to_swap_bytes(self, file):

    from numpy import fromstring, int32
    v = fromstring(file.read(4), int32)[0]
    file.seek(0,0)
    return (v < 0 or v >= 65536)
    
  # ---------------------------------------------------------------------------
  #
  def read_record(self, file, swap, skip = False):

    from numpy import int32
    size = string_values(file.read(4), int32, swap)[0]
    if size < 0:
      raise SyntaxError, 'Negative record size %d' % size
    if size > self.file_size:
      raise SyntaxError, ('Record size %d > file size %d'
                          % (size, self.file_size))

    if skip:
      file.seek(size, 1)
      string = ''
    else:
      string = file.read(size)

    from numpy import int32
    esize = string_values(file.read(4), int32, swap)[0]
    if esize != size:
      raise SyntaxError, ('Record size at end of record %d' % esize + 
                          ' != size at head of record %d' % size)
      
    return string
    
  # ---------------------------------------------------------------------------
  #
  def skip_record(self, file, swap):

    self.read_record(file, swap, skip = True)
    
  # ---------------------------------------------------------------------------
  #
  def matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import read_array
    data = read_array(self.path, self.data_offset + 4,
                      ijk_origin, ijk_size, ijk_step, self.size,
                      self.value_type, self.swap, progress)
    return data

# -----------------------------------------------------------------------------
#
def string_values(string, type, swap):

  from numpy import fromstring
  values = fromstring(string, type)
  if swap:
    values = values.byteswap()
  return values
