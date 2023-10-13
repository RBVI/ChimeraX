# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def save_image(session, path, format_name=None, width=None, height=None,
               supersample=3, pixel_size=None, transparent_background=False,
               quality=95):
    '''
    Save an image of the current graphics window contents.
    '''
    from chimerax.core.errors import UserError, LimitationError
    has_graphics = session.main_view.render is not None
    if not has_graphics:
        raise LimitationError("Unable to save images because OpenGL rendering is not available")
    from os.path import dirname, exists
    dir = dirname(path)
    if dir and not exists(dir):
        raise UserError('Directory "%s" does not exist' % dir)

    if pixel_size is not None:
        if width is not None or height is not None:
            raise UserError('Cannot specify width or height if pixel_size is given')
        v = session.main_view
        b = v.drawing_bounds()
        if b is None:
            raise UserError('Cannot specify use pixel_size option when nothing is shown')
        psize = v.pixel_size(b.center())
        if psize > 0 and pixel_size > 0:
            f = psize / pixel_size
            w, h = v.window_size
            from math import ceil
            width, height = int(ceil(f * w)), int(ceil(f * h))
        else:
            raise UserError('Pixel size option (%g) and screen pixel size (%g) must be positive'
                            % (pixel_size, psize))

    from chimerax.core.session import standard_metadata
    std_metadata = standard_metadata()
    metadata = {}
    if format_name is None:
        format_name = _format_name_from_suffix(path)
        if format_name is None:
            raise UserError('save_image(): File suffix not a known image format, %s (require %s)' % (path, ', '.join(_suffix_formats.keys())))
    if format_name == 'PNG':
        metadata['optimize'] = True
        # if dpi is not None:
        #     metadata['dpi'] = (dpi, dpi)
        if session.main_view.render.opengl_context.pixel_scale() == 2:
            metadata['dpi'] = (144, 144)
        from PIL import PngImagePlugin
        pnginfo = PngImagePlugin.PngInfo()
        # tags are from <https://www.w3.org/TR/PNG/#11textinfo>

        def add_text(keyword, value):
            try:
                b = value.encode('latin-1')
            except UnicodeEncodeError:
                pnginfo.add_itxt(keyword, value)
            else:
                pnginfo.add_text(keyword, b)
        # add_text('Title', description)
        add_text('Creation Time', std_metadata['created'])
        add_text('Software', std_metadata['generator'])
        add_text('Author', std_metadata['creator'])
        add_text('Copy' 'right', std_metadata['dateCopyrighted'])
        metadata['pnginfo'] = pnginfo
    elif format_name == 'TIFF':
        # metadata['compression'] = 'lzw:2'
        # metadata['description'] = description
        metadata['software'] = std_metadata['generator']
        # TIFF dates are YYYY:MM:DD HH:MM:SS (local timezone)
        import datetime as dt
        metadata['date_time'] = dt.datetime.now().strftime('%Y:%m:%d %H:%M:%S')
        metadata['artist'] = std_metadata['creator']
        # TIFF copy right is ASCII, so no Unicode symbols
        cp = std_metadata['dateCopyrighted']
        if cp[0] == '\N{COPYRIGHT SIGN}':
            cp = 'Copy' 'right' + cp[1:]
        metadata['copy' 'right'] = cp
        # if units == 'pixels':
        #     dpi = None
        # elif units in ('points', 'inches'):
        #     metadata['resolution unit'] = 'inch'
        #     metadata['x resolution'] = dpi
        #     metadata['y resolution'] = dpi
        # elif units in ('millimeters', 'centimeters'):
        #     adjust = convert['centimeters'] / convert['inches']
        #     dpcm = dpi * adjust
        #     metadata['resolution unit'] = 'cm'
        #     metadata['x resolution'] = dpcm
        #     metadata['y resolution'] = dpcm
    elif format_name == 'JPEG':
        if transparent_background:
            raise UserError('The JPEG file format does not support transparency, use PNG or TIFF instead.')
        metadata['quality'] = quality
        # if dpi is not None:
        #     # PIL's jpeg_encoder requires integer dpi values
        #     metadata['dpi'] = (int(dpi), int(dpi))
        # TODO: create exif with metadata using piexif package?
        # metadata['exif'] = exif

    view = session.main_view
    view.render.make_current()
    max_size = view.render.max_framebuffer_size()
    if max_size and ((width is not None and width > max_size)
                     or (height is not None and height > max_size)):
        raise UserError('Image size %d x %d too large, exceeds maximum OpenGL render buffer size %d'
                        % (width, height, max_size))

    i = view.image(width, height, supersample=supersample,
                   transparent_background=transparent_background)
    if i is not None:
        try:
            i.save(path, format_name, **metadata)
        except PermissionError:
            from chimerax.core.errors import UserError
            raise UserError('Permission denied writing file %s' % path)
    else:
        msg = "Unable to save image"
        if width is not None:
            msg += ', width %d' % width
        if height is not None:
            msg += ', height %d' % height
        session.logger.warning(msg)

_suffix_formats = {'bmp':'BMP', 'gif':'GIF', 'jpg':'JPEG', 'jpeg':'JPEG',
                   'png':'PNG', 'ppm':'PPM', 'tif':'TIFF', 'tiff':'TIFF'}
def _format_name_from_suffix(path):
    i = path.rfind('.')
    if i < 0:
        return None
    suffix = path[i+1:]
    return _suffix_formats.get(suffix.lower())
