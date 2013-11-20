# -----------------------------------------------------------------------------
# Read Gaussian cube file.
#
# Documentation is in Gaussian 98 manual under Cube keyword.
#
# Example file:
#
# azurin 1JZF
# SCF Molecular Orbitals
# -141  -22.336829  -16.222711  -12.353810
#   95    0.436104    0.000000    0.000000
#   83    0.000000    0.436104    0.000000
#   65    0.000000    0.000000    0.436104
#    1    1.000000   12.122777   -2.902144   -5.146283
#    6    6.000000   12.104072   -2.113008   -3.229603
#    ... (removed additional 138 atom lines)
#    1    1.000000   12.112141   -5.890766   -0.172029
#    1 1515
# -1.84745E-19 -2.68150E-19 -3.85426E-19 -5.51630E-19 -7.94108E-19 -1.16861E-18
# -1.79519E-18 -2.93148E-18 -5.10836E-18 -9.35382E-18 -1.75136E-17 -3.26273E-17
# ... (removed 86734 additional lines containing density values)
#
# First two lines are descriptive text.
# Line 3 is number of atoms and origin.  Negative number of atoms means
#  file contains molecular orbitals described on an additional line following
#  atoms.
# Line 4-6, grid size along each axis, and axis vector.
# Following num_atoms lines: atomic number, charge, x, y, z
# One line for molecular orbitals if num_atoms on line 3 was negative:
#   (number of orbitals) followed by orbital numbers for each orbital
# Density values, at most 6 per line, space separated.
#
# Grid origin and step are in units of the Bohr radius.
#

bohr_radius = 0.5291772108      # Angstroms

# -----------------------------------------------------------------------------
#
class FileFormatError(Exception):
  pass

# -----------------------------------------------------------------------------
#
class Gaussian_Cube:

  def __init__(self, path):

    self.matrices = None
    
    self.path = path
    #
    # Open file in binary mode 'rb'.  Opening in mode 'r' in Python 2.4.2
    # on Windows with '\n' line endings gives incorrect f.tell() values,
    #
    f = open(path, 'rb')

    self.title = f.readline()
    self.description = f.readline()

    if3_line = (int, float, float, float)
    ao = read_line_values(f.readline(), if3_line, 'atom count / origin')
    num_atoms = abs(ao[0])
    have_orbitals = (ao[0] < 0)
    self.origin = tuple(map(lambda o: bohr_radius * o, ao[1:]))
    
    grid_size = []
    step = []
    grid_axes = []
    for axis in range(3):
      sa = read_line_values(f.readline(), if3_line, 'grid size / axis')
      grid_size.append(sa[0])
      axis = sa[1:]
      n = norm(axis)
      step.append(n)
      grid_axes.append(map(lambda x: x/n, axis))
    self.grid_size = tuple(grid_size)
    self.step = tuple(map(lambda s: bohr_radius * s, step))
    self.grid_axes = tuple(map(tuple, grid_axes))

    if4_line = (int, float, float, float, float)
    atoms = []
    for a in range(num_atoms):
      ncxyz = read_line_values(f.readline(), if4_line, 'atom position')
      atoms.append(ncxyz)
    self.atoms = atoms

    if have_orbitals:
      fields = f.readline().split()
      self.orbital_numbers = fields[1:]
      self.component_names = self.orbital_numbers
    else:
      self.component_names = ['']
    self.num_components = len(self.component_names)

    self.data_offset = f.tell()
    
    f.close()

  # ---------------------------------------------------------------------------
  #
  def matrix(self, component_number, progress):

    if self.matrices == None:
      size = self.grid_size + (self.num_components,)
      from VolumeData.readarray import read_text_floats
      self.matrices = read_text_floats(self.path, self.data_offset, size,
                                       transpose = True, progress = progress)
    return self.matrices[component_number]
  
# -----------------------------------------------------------------------------
#
def norm(v):

  import math
  n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
  return n
  
# -----------------------------------------------------------------------------
# Read ascii numeric values on a line.
#
def read_line_values(line, types, descrip):

  try:
    fields = line.split()[:len(types)]
    values = map(lambda t, f: t(f), types, fields)
  except:
    raise FileFormatError('Error parsing %s on line:\n %s' % (descrip, line))
  return values
