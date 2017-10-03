# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def imagej_grids(path):

    pi = imagej_pixels(path)
    grids = [ImageJ_Grid(pi, c) for c in range(pi.nchannels)]
    return grids

# -----------------------------------------------------------------------------
#
from .. import Grid_Data
class ImageJ_Grid(Grid_Data):

  def __init__(self, imagej_pixels, channel):

    self.imagej_pixels = d = imagej_pixels
    self.initial_style = 'solid'

    name = d.name
    if d.nchannels > 1:
        name += ' ch%d' % channel

    from .ome_tiff import default_channel_colors
    rgba = default_channel_colors[channel % len(default_channel_colors)]

    origin = (0,0,0)
    Grid_Data.__init__(self, d.grid_size, d.value_type,
                       origin, d.grid_spacing,
                       name = name, path = d.path,
                       file_type = 'imagestack',
                       channel = channel,
                       default_color = rgba)

        
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import allocate_array
    array = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    dsize = self.size
    from numpy import zeros
    ia = zeros((dsize[1],dsize[0]), self.value_type)
    ia_1d = ia.ravel()
    c = self.channel
    pi = self.imagej_pixels
    for k in range(k0, k0+ksz, kstep):
      if progress:
        progress.plane((k-k0)//kstep)
      pi.plane_data(c, k, ia_1d)
      array[(k-k0)//kstep,:,:] = ia[j0:j0+jsz:jstep,i0:i0+isz:istep]
    return array

# -----------------------------------------------------------------------------
#  Example ImageJ TIFF description tag:
#
# "ImageJ=1.51n
#  images=70
#  channels=2
#  slices=35
#  hyperstack=true
#  mode=composite
#  unit=micron
#  spacing=0.125
#  loop=false
#  min=0.0
#  max=12441.318359375
# "
#
def imagej_pixels(path):

    X_RESOLUTION_TAG = 282
    Y_RESOLUTION_TAG = 283
    DESCRIPTION_TAG = 270
    
    from PIL import Image
    i = Image.open(path)

    pixel_width = 1.0
    if X_RESOLUTION_TAG in i.tag:
        v = i.tag[X_RESOLUTION_TAG]
        num,denom = v[0]
        if num != 0:
            pixel_width = denom/num	# 1/value

    pixel_height = 1.0
    if Y_RESOLUTION_TAG in i.tag:
        v = i.tag[Y_RESOLUTION_TAG]
        num,denom = v[0]
        if num != 0:
            pixel_height = denom/num	# 1/value

    pixel_size = (pixel_width, pixel_height)

    header = None
    for d in i.tag[DESCRIPTION_TAG]:
        if d.startswith('ImageJ='):
            header = d
            break

    if header is None:
        from os.path import basename
        raise TypeError('ImageJ TIFF file %s does not have an image description tag'
                        ' starting with "ImageJ=<version>"' % basename(path))
        
    h = {}
    lines = header.split('\n')
    for line in lines:
        kv = line.split('=')
        if len(kv) == 2:
            h[kv[0]] = kv[1]

    from os.path import basename
    name = basename(path)
    xsize, ysize = i.size
    zsize = int(h['slices'])
    grid_size = (xsize, ysize, zsize)
    zspacing = float(h['spacing']) if 'spacing' in h else 1
    grid_spacing = (pixel_width, pixel_height, zspacing)
    nc = int(h['channels']) if 'channels' in h else 1
    from . import imagestack_format
    value_type = imagestack_format. pillow_numpy_value_type(i.mode)

    pi = ImageJ_Pixels(path, name, value_type, grid_size, grid_spacing, nc)
    return pi

# -----------------------------------------------------------------------------
#
class ImageJ_Pixels:
    def __init__(self, path, name, value_type, grid_size, grid_spacing, nchannels):
        self.path = path
        self.name = name
        self.value_type = value_type	# Numpy dtype
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.nchannels = nchannels
        self.image = None
        self._last_plane = 0

    def plane_data(self, channel, k, pixel_values):
        plane = self.nchannels*k + channel
        im = self.image_plane(plane)
        from numpy import array
        pixel_values[:] = array(im).ravel()

    def image_plane(self, plane):
        im = self.image
        if im is None or plane < self._last_plane:
            from PIL import Image
            self.image = im = Image.open(self.path)
        self._last_plane = plane
        
        im.seek(plane)

        return im
