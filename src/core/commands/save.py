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
    from .. import io

    if format is None:
        from .. import io
        fmt, fname, compress = io.deduce_format(filename, savable = True)
    else:
        format = format.casefold()
        fmt = format_from_short_name(format)
        if format_name is None:
            fnames = sum([tuple(f.short_names) for f in io.formats()], ())
            from ..errors import UserError
            raise UserError("Unrecognized format '%s', must be one of %s" %
                            (format, ', '.join(fnames)))
        if fmt.export_func is None:
            from ..errors import UserError
            raise UserError("Format '%s' cannot be saved." % format)
    
    from os.path import splitext
    suffix = splitext(filename)[1][1:].casefold()
    if not suffix and fmt:
        suffix = fmt.extensions[0]
        filename += suffix

    save_func = fmt.export_func
    if save_func is None:
        suffixes = ', '.join(sum([f.extensions for f in io.formats() if f.export_func], []))
        from ..errors import UserError
        if not suffix:
            msg = 'Missing file suffix, require one of %s' % suffixes
        else:
            msg = 'Unrecognized file suffix "%s", require one of %s' % (suffix, suffixes)
        raise UserError(msg)

    kw = {
        'format': format,
        'width': width,
        'height': height,
        'supersample': supersample,
        'pixel_size': pixel_size,
        'transparent_background': transparent_background,
        'quality': quality,
    }
    
    save_func(session, filename, **kw)


def register_command(session):
    from . import CmdDesc, register, EnumOf, SaveFileNameArg, DynamicEnum
    from . import IntArg, BoolArg, PositiveIntArg, Bounded, FloatArg
    from .. import session as ses

    def save_formats():
        from .. import io
        names = sum((tuple(f.short_names) for f in io.formats() if f.export_func), ())
        return names

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
            ('format', DynamicEnum(save_formats)),
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

    from ..image import image_formats
    img_fmts = EnumOf([f.name for f in image_formats])
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
    from ..image import save_image
    register('save image', desc, save_image)
