# -----------------------------------------------------------------------------
# Read MacMolPlt "3D surface" format.
#
# Documentation is in MacMolPlt 7.2.1 manual under Files.
#
# Example file:
#
# 3D Total Electron Density Visible
# 99 27 104   //nx ny nz
# -7.76729 -2.35589 -8.66586   //Origin of the 3D grid
# 0.180527 0.17992 0.180192   //x increment, y inc, z inc/ grid(x(y(z)))
# 1.4013e-44 1.00893e-43 ...
#  (nx*ny*nz grid values)
#
# Units are usually in Angstroms.
#

# -----------------------------------------------------------------------------
#
class MacMolPlt_Data:

  def __init__(self, path):

    self.data = None
    
    self.path = path

    import sys
    if sys.platform == 'win32':
      # Open file in binary mode 'rb'.  Opening in mode 'r' in Python 2.4.2
      # on Windows with '\n' line endings gives incorrect f.tell() values,
      mode = 'rb'
    else:
      # Opening file in mode 'rb' on mac with Python 2.5.2 and '\r' line
      # endings reads full file as a single line.
      mode = 'rU'
    f = open(path, mode)

    self.title = f.readline()
    self.grid_size = read_line_values(f.readline(), (int, int, int), 'nx,ny,nz')
    self.origin = read_line_values(f.readline(), (float,float,float), 'origin')
    self.step = read_line_values(f.readline(), (float,float,float), 'step')

    self.data_offset = f.tell()
    
    f.close()

  # ---------------------------------------------------------------------------
  #
  def matrix(self, progress):

    if self.data is None:
      from ..readarray import read_text_floats
      self.data = read_text_floats(self.path, self.data_offset, self.grid_size,
                                   transpose = True, progress = progress)
    return self.data
  
# -----------------------------------------------------------------------------
# Read ascii numeric values on a line.
#
def read_line_values(line, types, descrip):

  try:
    fields = line.split()[:len(types)]
    values = map(lambda t, f: t(f), types, fields)
  except:
    raise SyntaxError, 'Error parsing %s on line:\n %s' % (descrip, line)
  return values
