# -----------------------------------------------------------------------------
# Read PROFEC potential energy grid file.
#
# From Amber7 manual page 247:
# 
# Data formats
# 
#     vdw, esp format:
# 
# 
#     ## header and comments
#     ##
#     ## input, output files
#     ##
#     ## control variables from
#     ##
#     ## included, excluded atoms
#     ##
#     ## atom type information from
#     ##
#     ##
#     21 21 21 # number of Grid spaces (3I12)
#     .500 .500 .500 # size of Grid spacing (3F12.3)
#     -.109328 .993533 -.030654 # rotation matrix (3F12.6)
#     .900871 .112071 .419371 # rotation matrix (3F12.6)
#     .420094 .018234 -.907297 # rotation matrix (3F12.6)
#     51.064120 30.547364 24.991392 # translation vector (3F12.6)
#     .92329E+01 .54755E+01 .54026E+01 .57455E+01 .14020E+02 .34699E+02
#
# Data matrix values are assumed to be in format 6E12.6, so spaces need not
# be present between data values.
#

# -----------------------------------------------------------------------------
#
class FileFormatError(Exception):
  pass

# -----------------------------------------------------------------------------
#
class PROFEC_Potential:

  def __init__(self, path):

    self.path = path
    #
    # Open file in binary mode 'rb'.  Opening in mode 'r' in Python 2.4.2
    # on Windows with '\n' line endings gives incorrect f.tell() values,
    #
    f = open(path, 'rb')

    line = skip_comment_lines(f, '#')

    self.grid_size = read_line_values(line, 3, int, 'grid dimensions')
    self.step = read_line_values(f.readline(), 3, float, 'grid spacing')
    rot1 = read_line_values(f.readline(), 3, float, 'rotation matrix row 1')
    rot2 = read_line_values(f.readline(), 3, float, 'rotation matrix row 2')
    rot3 = read_line_values(f.readline(), 3, float, 'rotation matrix row 3')
    self.rotation = (rot1, rot2, rot3)
    self.origin = read_line_values(f.readline(), 3, float, 'translation vector')

    self.data_offset = f.tell()
    
    f.close()

  # ---------------------------------------------------------------------------
  #
  def matrix(self, progress):

    from VolumeData.readarray import read_text_floats
    data = read_text_floats(self.path, self.data_offset, self.grid_size,
                            progress = progress, line_format = (12,6))
    return data
  
# -----------------------------------------------------------------------------
#
def skip_comment_lines(f, comment_character):

  while True:
    line = f.readline()
    if line == '' or line[0] != comment_character:
      break
  return line
  
# -----------------------------------------------------------------------------
# Read ascii numeric values on a line.
#
def read_line_values(line, count, type, descrip):

  try:
    values = map(type, line.split()[:count])
  except:
    raise FileFormatError('Error parsing %s on line:\n %s' % (descrip, line))
  return values
