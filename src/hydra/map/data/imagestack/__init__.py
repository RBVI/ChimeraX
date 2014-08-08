# -----------------------------------------------------------------------------
# Image stack file reader.
#
def open(paths):

  from .imagestack_grid import Image_Stack_Grid
  from .imagestack_format import multipage_image
  if multipage_image(paths[0]):
    return [Image_Stack_Grid([p]) for p in paths]
  return [Image_Stack_Grid(paths)]
