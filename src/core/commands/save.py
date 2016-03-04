# vim: set expandtab shiftwidth=4 softtabstop=4:


def save(session, filename, width=None, height=None, supersample=3,
         pixel_size=None, transparent_background=False, quality=95, format=None):
    '''Save data, sessions, images.

    Parameters
    ----------
    filename : string
        File to save.
        File suffix determines what type of file is saved unless the format option is given.
        For sessions the suffix is .cxs.
        Image files can be saved with .png, .jpg, .tif, .ppm, .gif suffixes.
    width : integer
        Width of image in pixels for saving image files.
        If width is not specified the current graphics window width is used.
    height : integer
        Height of image in pixels for saving image files.
    pixel_size : float
        The size of one pixel in the saved image in physical units,
        typically Angstroms.  Set the image width and height in pixels is set
        to achieve this physical pixel size.  For perspective projection the
        pixel size is at the depth of the center of the bounding box of the
        visible scene.
    supersample : integer
        Supersampling for saving images.
        Makes an image N times larger in each dimension
        then averages to requested image size to produce smoother object edges.
        Values of 2 or 3 are most useful,
        with little improvement for larger values.
    transparent_background : bool
        Save image with transparent background.
    format : string
        Recognized formats are session, or for saving images png, jpeg, tiff, gif, ppm, bmp.
        If not specified, then the filename suffix is used to identify the format.
    '''
    if format is not None:
        format = format.casefold()
        if format not in format_suffix:
            from ..errors import UserError
            raise UserError("Unrecognized format '%s', must be one of %s" %
                            (format, ', '.join(format_suffix.keys())))
    from os.path import splitext
    suffix = splitext(filename)[1][1:].casefold()
    if not suffix and format:
        suffix = format_suffix[format]
        filename += '.%s' % suffix
    if suffix in pil_image_formats:
        save_image(session, filename, format, width, height,
                   supersample, pixel_size, transparent_background, quality)
    elif suffix == format_suffix['session']:
        from ..session import save as save_session
        save_session(session, filename)
    else:
        from ..errors import UserError
        from . import commas
        suffixes = commas(["'%s'" % i for i in format_suffix.values()])
        if not suffix:
            raise UserError('Missing file suffix, require one of %s' % suffixes)
        raise UserError('Unrecognized file suffix "%s", require one of %s' %
                        (suffix, suffixes))

# Map format name used by save command to file suffix.
from ..session import SESSION_SUFFIX
format_suffix = {
    'session': SESSION_SUFFIX[1:],
}
image_format_suffix = {
    'png': 'png',
    'jpeg': 'jpg',
    'tiff': 'tif',
    'gif': 'gif',
    'ppm': 'ppm',
    'bmp': 'bmp',
}
format_suffix.update(image_format_suffix)

def register_command(session):
    from . import CmdDesc, register, EnumOf, SaveFileNameArg
    from . import IntArg, BoolArg, PositiveIntArg, Bounded, FloatArg
    from .. import session as ses
    ses_suffix = ses.SESSION_SUFFIX[1:]
    img_fmts = EnumOf(image_format_suffix.keys())
    all_fmts = EnumOf(format_suffix.keys())
    quality_arg = Bounded(IntArg, min=0, max=100)
    desc = CmdDesc(
        required=[('filename', SaveFileNameArg), ],
        keyword=[
            ('width', PositiveIntArg),
            ('height', PositiveIntArg),
            ('supersample', PositiveIntArg),
            ('pixel_size', FloatArg),
            ('transparent_background', BoolArg),
            ('quality', quality_arg),
            ('format', all_fmts),
        ],
        synopsis='save session or image'
    )
    register('save', desc, save)

    desc = CmdDesc(
        required=[('filename', SaveFileNameArg), ],
        # synopsis='save session'
    )
    from .. import session as ses
    register('save session', desc, ses.save)

    desc = CmdDesc(
        required=[('filename', SaveFileNameArg), ],
        keyword=[
            ('width', PositiveIntArg),
            ('height', PositiveIntArg),
            ('supersample', PositiveIntArg),
            ('pixel_size', FloatArg),
            ('transparent_background', BoolArg),
            ('quality', quality_arg),
            ('format', img_fmts),
        ],
        # synopsis='save image'
    )
    register('save image', desc, save_image)

# Map image file suffix to Pillow image format.
pil_image_formats = {
    'png': 'PNG',
    'jpg': 'JPEG',
    'tif': 'TIFF',
    'gif': 'GIF',
    'ppm': 'PPM',
    'bmp': 'BMP',
}

def save_image(session, filename, format=None, width=None, height=None,
               supersample=3, pixel_size=None, transparent_background=False, quality=95):
    '''
    Save an image of the current graphics window contents.
    '''
    from os.path import expanduser, dirname, exists, splitext
    path = expanduser(filename)         # Tilde expansion
    dir = dirname(path)
    if dir and not exists(dir):
        from ..errors import UserError
        raise UserError('Directory "%s" does not exist' % dir)

    if pixel_size is not None:
        from ..errors import UserError
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
            width, height = int(round(f*w)), int(round(f*h))
        else:
            raise UserError('Pixel size option (%g) and screen pixel size (%g) must be positive'
                            % (pixel_size, psize))
        
    suffix = splitext(path)[1][1:].casefold()
    if suffix == '':
        if format is None:
            suffix = 'png'
            path += '.' + suffix
        else:
            path += '.' + format_suffix[format]
    elif suffix not in pil_image_formats:
        raise UserError('Unrecognized image file suffix "%s"' % format)

    view = session.main_view
    i = view.image(width, height, supersample=supersample,
                   transparent_background=transparent_background)
    iformat = pil_image_formats[suffix if format is None else format_suffix[format]]
    i.save(path, iformat, quality=quality)
