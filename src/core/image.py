# vim: set expandtab shiftwidth=4 softtabstop=4:

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

class ImageFormat:
    def __init__(self, name, suffixes, pil_name):
        self.name = name
        self.suffixes = suffixes
        self.pil_name = pil_name

_formats = [
    ('png', ['png'], 'PNG'),
    ('jpeg', ['jpg','jpeg'],'JPEG'),
    ('tiff', ['tif','tiff'],'TIFF'),
    ('gif', ['gif'], 'GIF'),
    ('ppm', ['ppm'], 'PPM'),
    ('bmp', ['bmp'], 'BMP'),
]
default_format = 'png'
image_formats = [ImageFormat(name, suffixes, pil_name)
                 for name, suffixes, pil_name in _formats]

def save_image(session, filename, format, width=None, height=None,
               supersample=3, pixel_size=None, transparent_background=False, quality=95):
    '''
    Save an image of the current graphics window contents.
    '''
    from os.path import expanduser, dirname, exists, splitext
    path = expanduser(filename)         # Tilde expansion
    dir = dirname(path)
    if dir and not exists(dir):
        from .errors import UserError
        raise UserError('Directory "%s" does not exist' % dir)

    if pixel_size is not None:
        if width is not None or height is not None:
            from .errors import UserError
            raise UserError('Cannot specify width or height if pixel_size is given')
        v = session.main_view
        b = v.drawing_bounds()
        if b is None:
            from .errors import UserError
            raise UserError('Cannot specify use pixel_size option when nothing is shown')
        psize = v.pixel_size(b.center())
        if psize > 0 and pixel_size > 0:
            f = psize / pixel_size
            w, h = v.window_size
            width, height = int(round(f*w)), int(round(f*h))
        else:
            from .errors import UserError            
            raise UserError('Pixel size option (%g) and screen pixel size (%g) must be positive'
                            % (pixel_size, psize))

    fmt = None
    if format is not None:
        for f in image_formats:
            if f.name == format:
                fmt = f
        if fmt is None:
            from .errors import UserError
            raise UserError('Unknown image file format "%s"' % format)
        
    suffix = splitext(path)[1][1:].casefold()
    if suffix == '':
        if fmt is None:
            fmt = default_format
            path += '.' + default_format.suffixes[0]
        else:
            path += '.' + fmt.suffixes[0]
    elif fmt is None:
        for f in image_formats:
            if suffix in f.suffixes:
                fmt = f
        if fmt is None:
            from .errors import UserError
            raise UserError('Unknown image file suffix "%s"' % suffix)

    view = session.main_view
    i = view.image(width, height, supersample=supersample,
                   transparent_background=transparent_background)
    i.save(path, fmt.pil_name, quality=quality)

def register_image_save(session):
    from .io import register_format
    for format in image_formats:
        register_format("%s image" % format.name,
                        category = 'Image',
                        extensions = ['.%s' % s for s in format.suffixes],
                        nicknames=[format.name.casefold()],
                        export_func=save_image)

    # Register save command keywords for images
    from .commands import PositiveIntArg, FloatArg, BoolArg, Bounded, IntArg, add_keyword_arguments
    save_image_args = [
        ('width', PositiveIntArg),
        ('height', PositiveIntArg),
        ('supersample', PositiveIntArg),
        ('pixel_size', FloatArg),
        ('transparent_background', BoolArg),
        ('quality', Bounded(IntArg, min=0, max=100)),
    ]
    add_keyword_arguments('save', dict(save_image_args))

    # Register save image subcommand
    from .commands import CmdDesc, register, SaveFileNameArg
    from .commands.save import SaveFileFormatsArg, save
    desc = CmdDesc(
        required=[('filename', SaveFileNameArg)],
        keyword=[('format', SaveFileFormatsArg('Image'))] + save_image_args,
        synopsis='save image'
    )
    register('save image', desc, save, logger=session.logger)
