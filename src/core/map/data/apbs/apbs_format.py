# -----------------------------------------------------------------------------
# Read an Adaptive Poisson-Boltzmann Solver (APBS) electrostatics opendx file.
#

# -----------------------------------------------------------------------------
#
class APBS_Data:

  def __init__(self, path):

    self.path = path

    #
    # Open file in binary mode 'rb'.  Opening in mode 'r' in Python 2.4.2
    # on Windows with '\n' line endings gives incorrect f.tell() values,
    #
    f = open(path, 'rb')

    (self.comments,
     self.grid_size,
     self.xyz_origin,
     self.xyz_step,
     self.data_offset) = self.read_header(f)

    f.close()

  # ---------------------------------------------------------------------------
  # Example file header:
  #
  # # Data from APBS 0.3.2
  # # 
  # # POTENTIAL (kT/e)
  # # 
  # object 1 class gridpositions counts 65 97 97
  # origin -8.064450e+00 -4.768105e+01 -4.266980e+01
  # delta 3.779047e-01 0.000000e+00 0.000000e+00
  # delta 0.000000e+00 4.098240e-01 0.000000e+00
  # delta 0.000000e+00 0.000000e+00 4.099125e-01
  # object 2 class gridconnections counts 65 97 97
  # object 3 class array type double rank 0 items 611585 data follows
  # 8.921700e-01 9.000006e-01 9.078676e-01 
  # 9.157664e-01 9.236924e-01 9.316403e-01 
  # ...
  #
  def read_header(self, f):

    comments = []
    grid_size = None
    xyz_origin = None
    xyz_step = []
    data_offset = None
    
    while True:
      line = f.readline()
      if not line:
        break
      sline = line.strip()

      if len(sline) == 0:
        raise SyntaxError('Incorrect format: blank header line.  Comments must begin with # character.')
      elif sline[0] == '#':
        comments.append(line)
      elif sline.startswith(b'object 1 class gridpositions counts '):
        fields = sline.split()
        if len(fields) < 8:
          raise SyntaxError('Less than 8 fields in line: %s' % line)
        try:
          grid_size = tuple(int(s) for s in fields[5:8])
        except ValueError:
          raise SyntaxError('Fields 6-8 are not integers in line: %s' % line)
      elif sline.startswith(b'origin '):
        fields = sline.split()
        if len(fields) < 4:
          raise SyntaxError('Less than 4 fields in line: %s' % line)
        try:
          xyz_origin = tuple(float(o) for o in fields[1:4])
        except ValueError:
          raise SyntaxError('Fields 2-5 are not floats in line: %s' % line)
      elif sline.startswith(b'delta '):
        fields = sline.split()
        if len(fields) < 4:
          raise SyntaxError('Less than 4 fields in line: %s' % line)
        try:
          delta = tuple(float(d) for d in fields[1:4])
        except ValueError:
          raise SyntaxError('Fields 2-5 are not floats in line: %s' % line)
        n = len(xyz_step)
        if n == 3:
          raise SyntaxError('More than 3 lines starting with "delta"')
        xyz_step.append(delta[n])
      elif sline.endswith(b'data follows'):
        data_offset = f.tell()
        break

    xyz_step = tuple(xyz_step)
    
    if grid_size == None:
      raise SyntaxError('Missing "object 1 class gridpositions counts <l> <m> <n>" header line')
    if xyz_origin == None:
      raise SyntaxError('Missing "origin" header line')
    if len(xyz_step) != 3:
      raise SyntaxError('Only found %d "delta" lines, require 3' % len(xyz_step))
    if data_offset == None:
      raise SyntaxError('Missing header line ending with "data follows"')

    return comments, grid_size, xyz_origin, xyz_step, data_offset

  # ---------------------------------------------------------------------------
  #
  def matrix(self, progress):

    from ..readarray import read_text_floats
    data = read_text_floats(self.path, self.data_offset, self.grid_size,
                            transpose = True, progress = progress)
    return data
