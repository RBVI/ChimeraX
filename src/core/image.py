# vim: set expandtab shiftwidth=4 softtabstop=4:

class ImageFormat:
    def __init__(self, name, suffix, pil_name):
        self.name = name
        self.suffix = suffix
        self.pil_name = pil_name

_formats = [
    ('png', 'png', 'PNG'),
    ('jpeg', 'jpg','JPEG'),
    ('tiff', 'tif','TIFF'),
    ('gif', 'gif', 'GIF'),
    ('ppm', 'ppm', 'PPM'),
    ('bmp', 'bmp', 'BMP'),
]
default_format = 'png'
image_formats = [ImageFormat(name, suffix, pil_name)
                 for name, suffix, pil_name in _formats]

def save_image(session, filename, format=None, width=None, height=None,
               supersample=3, pixel_size=None, transparent_background=False, quality=95, **kw):
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
            path += '.' + default_format.suffix
        else:
            path += '.' + fmt.suffix
    elif fmt is None:
        for f in image_formats:
            if f.suffix == suffix:
                fmt = f
        if fmt is None:
            from .errors import UserError
            raise UserError('Unknown image file suffix "%s"' % suffix)

    view = session.main_view
    i = view.image(width, height, supersample=supersample,
                   transparent_background=transparent_background)
    i.save(path, fmt.pil_name, quality=quality)

def register_image_save():
    from .io import register_format
    for format in image_formats:
        register_format("%s image" % format.name,
                        category = 'Image',
                        extensions = ['.%s' % format.suffix],
                        short_names = [format.name],
                        export_func=save_image)
