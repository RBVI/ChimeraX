# -----------------------------------------------------------------------------
# Image stack file reader.
#
def open(paths):

  from .imagestack_grid import Image_Stack_Grid
  return [Image_Stack_Grid(paths)]
