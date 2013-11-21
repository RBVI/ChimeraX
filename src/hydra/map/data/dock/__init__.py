# -----------------------------------------------------------------------------
# Docking grid files from DOCK program developed by Tack Kuntz's group at UCSF.
#
def open(path):

  from dock_grid import read_dock_file
  return read_dock_file(path)
