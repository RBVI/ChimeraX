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

# -----------------------------------------------------------------------------
# Save map as tiff with ImageJ meta data
#
def write_imagej_tiff(grid_data, path, options = {}, progress = None):
    if '{z}' in path:
        _write_2d_tiffs(grid_data, path)
    else:
        _write_3d_tiff(grid_data, path)
    
def _write_3d_tiff(grid_data, path):
    array = grid_data.matrix()
    # ImageJ stack must have axis order TZCYX.  So insert a channel dimension.
    zsize,ysize,xsize = array.shape
    channels = 1
    array_zcyx = array.reshape((zsize, channels, ysize, xsize))
    xstep, ystep, zstep = grid_data.step
    from tifffile import imwrite
    imwrite(path, array_zcyx, imagej=True, resolution=(1/xstep, 1/ystep),
            photometric='minisblack', metadata={'spacing': zstep, 'axes': 'ZCYX'})

def _write_2d_tiffs(grid_data, path):
    array = grid_data.matrix()
    xstep, ystep, zstep = grid_data.step
    zsize = array.shape[0]
    zformat = _z_suffix_format(zsize)
    from tifffile import imwrite
    for z in range(zsize):
        zpath = path.replace('{z}', zformat % z)
        imwrite(zpath, array[z], imagej=True, resolution=(1/xstep, 1/ystep),
                photometric='minisblack', metadata={'spacing': zstep, 'axes': 'YX'})

def _z_suffix_format(zsize):
    ndigits = 1
    while zsize >= 10**ndigits:
        ndigits += 1
    zformat = '%%0%dd' % ndigits
    return zformat
