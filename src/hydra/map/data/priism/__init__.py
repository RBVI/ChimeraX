# -----------------------------------------------------------------------------
# Priism microscope data file reader.
# Used by John Sedat and Dave Agaard groups at UCSF.
#
def open(path):

  from priism_grid import read_priism_file
  return read_priism_file(path)
