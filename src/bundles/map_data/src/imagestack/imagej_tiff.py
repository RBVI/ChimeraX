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
    ns = pi.ntimes * pi.nchannels * pi.grid_size[2]
    if ns > 1 and not pi.multiframe:
        from chimerax.core.errors import UserError
        raise UserError('ImageJ TIFF has %d slices (z = %d, nc = %d, nt = %d) but file as only 1 image.'
                        % (ns, pi.grid_size[2], pi.nchannels, pi.ntimes) +
                        '  Reader does not currently handle ImageJ packing of large data (> 4 Gbytes) as single plane.')

    nc = pi.nchannels * pi.ncolors
    if nc == 1 and pi.ntimes == 1:
        grids = [ImageJGrid(pi)]
    elif pi.ntimes == 1:
        grids = [ImageJGrid(pi, c) for c in range(nc)]
    else:
        grids = [ImageJGrid(pi, c, t) for c in range(nc) for t in range(pi.ntimes)]
    return grids

# -----------------------------------------------------------------------------
#
from .. import GridData
class ImageJGrid(GridData):

  def __init__(self, imagej_pixels, channel = None, time = None):

    self.imagej_pixels = d = imagej_pixels
    self.initial_style = 'image'

    name = d.name
    if d.nchannels > 1:
        name += ' ch%d' % channel

    grid_id = '' if channel is None else f'c{channel}'
    if time is not None:
        grid_id += f't{time}'
        
    origin = (0,0,0)
    GridData.__init__(self, d.grid_size, d.value_type,
                      origin, d.grid_spacing,
                      name = name, path = d.path,
                      file_type = 'imagestack',
                      grid_id = grid_id,
                      channel = channel,
                      time = time)

  # ---------------------------------------------------------------------------
  # Reading multiple planes at a time is twice as fast as one plane at a time
  # using tifffile.py so use read_matrix() instead of read_xy_plane().
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    c = self.channel
    if c is None:
        c = 0
    pi = self.imagej_pixels
    nc = pi.ncolors
    ch, cc = c // nc, c % nc
    klist = range(k0, k0+ksz, kstep)
    a = pi.planes_data(klist, cc, ch, self.time)
    array = a[:, j0:j0+jsz:jstep,i0:i0+isz:istep]
    return array

# -----------------------------------------------------------------------------
#  Example ImageJ TIFF description tag:
#
# "ImageJ=1.51n
#  images=70
#  channels=2
#  slices=35
#  frames=100
#  hyperstack=true
#  mode=composite
#  unit=micron
#  spacing=0.125
#  loop=false
#  min=0.0
#  max=12441.318359375
# "
#
# 2D images in XYCZT order.  slices = z size, channels = c size, frames = t size
#
def imagej_pixels(path):

    from tifffile import TiffFile, TIFF
    tif = TiffFile(path)
    pages = tif.pages
    page0 = pages[0]
    tags = page0.tags
    
    pixel_width = 1.0
    if 'XResolution' in tags:
        v = tags['XResolution'].value
        num,denom = v
        if num != 0:
            pixel_width = denom/num	# 1/value

    pixel_height = 1.0
    if 'YResolution' in tags:
        v = tags['YResolution'].value
        num,denom = v
        if num != 0:
            pixel_height = denom/num	# 1/value

    pixel_size = (pixel_width, pixel_height)

    header = None
    if 'ImageDescription' in tags:
        d = tags['ImageDescription'].value
        if d.startswith('ImageJ='):
            header = d

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
    shape = page0.shape
    ysize, xsize = shape[:2]
    zsize = int(h['slices']) if 'slices' in h else 1
    grid_size = (xsize, ysize, zsize)
    zspacing = float(h['spacing']) if 'spacing' in h else 1
    grid_spacing = (pixel_width, pixel_height, zspacing)
    nc = int(h['channels']) if 'channels' in h else 1
    nt = int(h['frames']) if 'frames' in h else 1
    value_type = page0.dtype
    ncolors = 3 if page0.photometric == TIFF.PHOTOMETRIC.RGB else 1
    multiframe = (zsize > 1 or nc > 1)

    # Check for ImageJ hyperstack format for > 4 GB data where TIFF has only one page.
    if zsize > 1 or nc > 1 or nt > 1:
        try:
            pages[1]
        except IndexError:
            from chimerax.core.errors import UserError
            raise UserError('Cannot read ImageJ hyperstack TIFF file "%s".  ImageJ TIFF files larger than 4 Gbytes do not follow the TIFF standard.  They include only one TIFF page and append the rest of the raw 2d images to the file.  ChimeraX cannot currently handle these hacked TIFF files.  Contact the ChimeraX developers and we can discuss adding support for this format.' % path)

    tif.close()
    
    pi = ImageJ_Pixels(path, name, value_type, grid_size, grid_spacing, ncolors, nc, nt, multiframe)
    return pi

# -----------------------------------------------------------------------------
#
class ImageJ_Pixels:
    def __init__(self, path, name, value_type, grid_size, grid_spacing,
                 ncolors, nchannels, ntimes, multiframe):
        self.path = path
        self.name = name
        self.value_type = value_type	# Numpy dtype
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.ncolors = ncolors		# 3 for RGB images, 1 for grayscale images
        self.nchannels = nchannels
        self.ntimes = ntimes
        self.multiframe = multiframe
        self.image = None
        self._last_plane = 0

    def planes_data(self, klist, color_component, channel, time):
        '''
        Read multiple planes using tifffile.py which is about 5 times
        faster than Pillow.
        '''
        nc = self.nchannels
        pbase = 0
        if channel is not None:
            pbase += channel
        if time is not None:
            nz = self.grid_size[2]
            pbase += nz*nc*time

        plist = [pbase + k for k in klist] if nc == 1 else [pbase + nc*k for k in klist]
        from tifffile import TiffFile
        with TiffFile(self.path) as tif:
            a = tif.asarray(key = plist)

        if self.ncolors > 1:
            if a.ndim == 4:
                a = a[:,:,:,color_component]
            elif a.ndim == 3 and len(plist) == 1:
                a = a[:,:,color_component].reshape((1,) + tuple(a.shape[:2]))
        elif a.ndim == 2:
            a = a.reshape((1,) + tuple(a.shape))	# Make single-plane 3d
            
        return a
        
    def plane_data(self, k, color_component, channel, time, pixel_values):
        '''Read single plane using Pillow.'''
        nc = self.nchannels
        plane = nc*k if nc > 1 else k
        if channel is not None:
            plane += channel
        if time is not None:
            nz = self.grid_size[2]
            plane += nz*nc*time
        im = self.image_plane(plane)
        from numpy import array
        a = array(im)
        if len(a.shape) == 3:
            a = a[:,:,color_component]
        pixel_values[:] = a.ravel()

    def image_plane(self, plane):
        im = self.image
        if im is None or plane < self._last_plane:
            from PIL import Image
            self.image = im = Image.open(self.path)
        self._last_plane = plane

        im.seek(plane)

        return im
