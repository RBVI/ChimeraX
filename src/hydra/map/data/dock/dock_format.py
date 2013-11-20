# -----------------------------------------------------------------------------
# Read DOCK (Tack Kuntz, UCSF) grid files containing bump map (.bmp),
# contact (.cnt), and energy (.nrg) scores.
# Byte swapping is done if needed.
#

# -----------------------------------------------------------------------------
#
class Dock_Data:

  def __init__(self, path):

    self.path = path

    file = open(path, 'rb')       # Raise exception if file cannot be opened.
    file.seek(0,2)
    file_size = file.tell()       # Determine file size
    file.close()

    from os.path import basename
    self.name = basename(path)

    if path.endswith('.bmp'):
      cnames = ['bump']
      expected_array_size = file_size - 32
    elif path.endswith('.cnt'):
      cnames = ['contact']
      expected_array_size = (file_size - 4) / 2
    elif path.endswith('.nrg'):
      cnames = ['vdw attract', 'vdw repel', 'electrostatic']
      expected_array_size = (file_size - 16) / 12
    else:
      raise SyntaxError, 'DOCK file %s suffix must be .bmp or .cnt or .nrg' % self.name
    self.component_names = cnames
    self.bump_path = path[:path.rfind('.')] + '.bmp'
    
    bmp_file = open(self.bump_path, 'rb')
    self.byte_swap = self.needs_byte_swap(bmp_file, expected_array_size)
    array_size, spacing, origin, axis_sizes = \
      self.read_bump_map_header(bmp_file)
    bmp_file.close()

    self.array_size = array_size        # number of array elements
    self.data_size = axis_sizes
    self.data_step = (spacing, spacing, spacing)
    self.data_origin = origin

  # -------------------------------------------------------------------------
  # Infer file byte order by comparing size of array and file size.
  #
  def needs_byte_swap(self, bmp_file, expected_array_size):

    pos = bmp_file.tell()

    bmp_file.seek(0,0)
    from numpy import fromstring, int32
    sa = fromstring(bmp_file.read(4), int32)   # returns array
    array_size = sa[0]

    bmp_file.seek(pos, 0)                   # go back to original position

    if array_size == expected_array_size:
      byte_swap = 0
    else:
      swapped_size = sa.byteswap()[0]
      if swapped_size == expected_array_size:
        byte_swap = 1
      else:
        raise SyntaxError, 'Expected array size based on DOCK file size is %d which does not agree with header array size %d or byte swapped header array size %d' % (expected_array_size, array_size, swapped_size)

    return byte_swap
    
  # ---------------------------------------------------------------------------
  #
  def read_bump_map_header(self, file):

    from numpy import int32, float32
    array_size = self.read_values(file, 1, int32)
    spacing = self.read_values(file, 1, float32)
    origin = self.read_values(file, 3, float32)
    axis_sizes = self.read_values(file, 3, int32)

    return array_size, spacing, origin, axis_sizes

  # ---------------------------------------------------------------------------
  #
  def read_values(self, file, count, type):

    from numpy import array, fromstring
    type_size = array([], type).itemsize

    a = fromstring(file.read(type_size*count), type)   # returns array
    if self.byte_swap:
      a = a.byteswap()

    if count == 1:
      values = a[0]
    else:
      values = tuple(a)

    return values

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, cname, ijk_origin, ijk_size, ijk_step, progress):

    from numpy import uint8, int16, float32, array
    array_specs = {
      'bump': (32, 0, uint8),
      'contact': (4, 0, int16),
      'vdw attract': (16, 0, float32),
      'vdw repel': (16, 1, float32),
      'electrostatic': (16, 2, float32),
      }

    offset, array_num, type = array_specs[cname]
    type_size = array([], type).itemsize
    array_bytes = self.array_size * type_size
    base = offset + array_num * array_bytes

    from VolumeData.readarray import read_array
    matrix = read_array(self.path, base, ijk_origin, ijk_size, ijk_step,
                        self.data_size, type, self.byte_swap, progress)

    return matrix

  # ---------------------------------------------------------------------------
  #
  def value_type(self, component_name):

    from numpy import uint8, int16, float32
    value_types = {
      'bump': uint8,
      'contact': int16,
      'vdw attract': float32,
      'vdw repel': float32,
      'electrostatic': float32,
      }
    return value_types[component_name]

  # ---------------------------------------------------------------------------
  #
  def color(self, component_name):

    colors = {
      'bump': (0,0,.75,1),              # blue
      'contact': (.5,.5,.75,1),         # gold
      'vdw attract': (0,.75,0,1),       # green
      'vdw repel': (.75,0,0,1),         # red
      'electrostatic': (0,.75,.75,1),   # cyan
      }
    return colors[component_name]
