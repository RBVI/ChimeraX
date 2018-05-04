# -----------------------------------------------------------------------------
# Read AmiraMesh file format map data.
#
class Amira_Mesh_Data:

  def __init__(self, path):

    self.path = path

    import os.path
    self.name = os.path.basename(path)
    
    file = open(path, 'rb')
    h = self.read_header(file)
    file.close()

    self.matrix_size = h['size']
    self.element_type = h['value_type']
    self.swap_bytes = False
    # TODO: not clear from docs if grid spacing requires divide by n or n-1.
    self.step = tuple((xmax-xmin)/n for xmin,xmax,n in zip(h['xyz_min'], h['xyz_max'], h['size']))
    self.data_offset = h['data_start']

  def read_header(self, file):
    
    line = file.readline()
    if not (line.startswith(b'# AmiraMesh BINARY-LITTLE-ENDIAN 2.1') or
            line.startswith(b'# Avizo BINARY-LITTLE-ENDIAN 2.1')):
      raise SyntaxError('First line of AmiraMesh file must start with '
                        '"# AmiraMesh BINARY-LITTLE-ENDIAN 2.1", '
                        'instead got "%s"' % line[:256])

    size = None
    bounds = None
    components = None
    dtype = None
    data_found = False
    start_data = False
    max_lines = 1000
    from numpy import float32, uint16
    for lc in range(max_lines):
      line = file.readline().strip()
      if line.startswith(b'define Lattice '):
        try:
          size = tuple(int(s) for s in line.split()[2:5])
        except ValueError:
          raise SyntaxError('Failed parsing integer values from line "%s"' % line[:256])
      elif line.startswith(b'BoundingBox '):
        try:
          bounds = tuple(float(x) for x in line.rstrip(b',').split()[1:7])
        except ValueError:
          raise SyntaxError('Failed parsing float values from line "%s"' % line[:256])
      elif line.startswith(b'Lattice { float['):
        try:
          components = int(line.split(b'[]')[1])
        except ValueError:
          raise SyntaxError('Failed parsing float count in line "%s"' % line[:256])
        dtype = float32
      elif line.startswith(b'Lattice { float'):
        components = 1
        dtype = float32
      elif line.startswith(b'Lattice { ushort['):
        try:
          components = int(line.split(b'[]')[1])
        except ValueError:
          raise SyntaxError('Failed parsing ushort count in line "%s"' % line[:256])
        dtype = uint16
      elif line.startswith(b'Lattice { ushort'):
        components = 1
        dtype = uint16
      elif line.startswith(b'# Data section follows'):
        data_found = True
      elif line.startswith(b'@1'):
        start_data = True
        break
    if size is None:
      raise SyntaxError('Did not find "define Lattice" mesh size line')
    elif len(size) != 3:
      raise SyntaxError('"define Lattice" specified %d values, expected 3' % (len(size),))
    elif bounds is None:
      raise SyntaxError('Did not find "BoundingBox" line')
    elif len(bounds) != 6:
      raise SyntaxError('"BoundingBox" specified %d values, expected 6' % (len(bounds),))
    elif components is None or dtype is None:
      raise SyntaxError('Did not find "Lattice { float ... }" line')
    elif components != 1:
      raise SyntaxError('Only handle single component data, got %d components' % (components,))
    elif not data_found:
      raise SyntaxError('Did not find "# Data section follows" line')
    elif not start_data:
      raise SyntaxError('Did not find "@1" line indicating start of data')

    data_start = file.tell()

    h = {
      'size': size,
      'xyz_min': (bounds[0], bounds[2], bounds[4]),
      'xyz_max': (bounds[1], bounds[3], bounds[5]),
      'components': components,
      'value_type': dtype,
      'data_start': data_start,
      }
    return h

  # ---------------------------------------------------------------------------
  # Reads a submatrix from a the file.
  # Returns 3d numpy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import read_array
    matrix = read_array(self.path, self.data_offset,
                        ijk_origin, ijk_size, ijk_step,
                        self.matrix_size, self.element_type, self.swap_bytes,
                        progress)
    return matrix
