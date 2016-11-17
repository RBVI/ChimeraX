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

def save(session, filename, models=None, format=None,
         width=None, height=None, supersample=3,
         pixel_size=None, transparent_background=False, quality=95,
         region = None, step = (1,1,1), mask_zone = True, chunk_shapes = None,
         append = None, compress = None, base_index = 1):
    '''Save data, sessions, images.

    Parameters
    ----------
    filename : string
        File to save.
        File suffix determines what type of file is saved unless the format option is given.
        For sessions the suffix is .cxs.
        Image files can be saved with .png, .jpg, .tif, .ppm, .gif suffixes.
    models : list of Model or None
        Models to save
    format : string
        Recognized formats are session, or for saving images png, jpeg, tiff, gif, ppm, bmp.
        If not specified, then the filename suffix is used to identify the format.
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
    region = None
    step = (1,1,1)
    mask_zone = True
    chunk_shapes = None,
    append = None
    compress = None
    base_index = 1
    '''
    from .. import io

    if format is None:
        from .. import io
        fmt, fname, compress = io.deduce_format(filename, open = False, save = True)
    else:
        format = format.casefold()
        from .open import format_from_name
        fmt = format_from_name(format, save=True, open=False)
        if fmt is None:
            fnames = sum([tuple(f.nicknames) for f in io.formats()], ())
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
        'models': models,
        'format': format,
        'width': width,
        'height': height,
        'supersample': supersample,
        'pixel_size': pixel_size,
        'transparent_background': transparent_background,
        'quality': quality,
        'region': region,
        'step': step,
        'mask_zone': mask_zone,
        'chunk_shapes': chunk_shapes,
        'append': append,
        'compress': compress,
        'base_index': base_index,
    }
    
    save_func(session, filename, **kw)

    if fmt.open_func and not fmt.name.endswith('image'):
        # Remember in file history
        from ..filehistory import remember_file
        remember_file(session, filename, fmt.nicknames[0], models or 'all models', file_saved = True)


def save_formats(session):
    '''Report file formats and suffixes that the save command knows about.'''
    if session.ui.is_gui:
        lines = ['<table border=1 cellspacing=0 cellpadding=2>', '<tr><th>File format<th>Short name(s)<th>Suffixes']
    else:
        session.logger.info('File format, Short name(s), Suffixes:')
    from .. import io
    from . import commas
    formats = list(f for f in io.formats() if f.export_func)
    formats.sort(key = lambda f: f.name)
    for f in formats:
        if session.ui.is_gui:
            lines.append('<tr><td>%s<td>%s<td>%s' % (f.name,
                commas(f.nicknames), ', '.join(f.extensions)))
        else:
            session.logger.info('    %s: %s: %s' % (f.name,
                commas(f.nicknames), ', '.join(f.extensions)))
    if session.ui.is_gui:
        lines.append('</table>')
        msg = '\n'.join(lines)
        session.logger.info(msg, is_html=True)

from . import DynamicEnum
class FileFormatArg(DynamicEnum):
    def __init__(self, category = None):
        DynamicEnum.__init__(self, self.formats)
        self.category = category
    def formats(self):
        cat = self.category
        from .. import io
        names = sum((tuple(f.nicknames) for f in io.formats()
                     if f.export_func and (cat is None or f.category == cat)),
                    ())
        return names
        
def register_command(session):
    from . import CmdDesc, register, EnumOf, SaveFileNameArg
    from . import IntArg, PositiveIntArg, Bounded, FloatArg, NoArg
    from . import ModelsArg, ListOf
    from ..map.mapargs import MapRegionArg, Int1or3Arg

    file_arg = [('filename', SaveFileNameArg)]
    models_arg = [('models', ModelsArg)]

    format_args = [('format', FileFormatArg())]
    from .. import toolshed
    map_format_args = [('format', FileFormatArg(toolshed.VOLUME))]
    image_format_args = [('format', FileFormatArg('Image'))]

    image_args = [
        ('width', PositiveIntArg),
        ('height', PositiveIntArg),
        ('supersample', PositiveIntArg),
        ('pixel_size', FloatArg),
        ('transparent_background', NoArg),
        ('quality', Bounded(IntArg, min=0, max=100))]

    map_args = [
        ('region', MapRegionArg),
        ('step', Int1or3Arg),
        ('mask_zone', NoArg),
        ('chunk_shapes', ListOf(EnumOf(('zyx','zxy','yxz','yzx','xzy','xyz')))),
        ('append', NoArg),
        ('compress', NoArg),
        ('base_index', IntArg)]

    desc = CmdDesc(
        required=file_arg,
        optional=models_arg,
        keyword=format_args + image_args + map_args,
        synopsis='save session or image'
    )
    register('save', desc, save)

    desc = CmdDesc(
        required=file_arg,
        synopsis='save session'
    )
    def save_session(session, filename, **kw):
        kw['format'] = 'ses'
        save(session, filename, **kw)
    register('save session', desc, save_session)

    desc = CmdDesc(
        required=file_arg,
        keyword=image_format_args + image_args,
        synopsis='save image'
    )
    def save_image(session, filename, **kw):
        save(session, filename, **kw)
    register('save image', desc, save_image)

    desc = CmdDesc(
        required=file_arg,
        optional=models_arg,
        keyword=map_format_args + map_args,
        synopsis='save map'
    )
    register('save map', desc, save)

    sf_desc = CmdDesc(synopsis='report formats that can be saved')
    register('save formats', sf_desc, save_formats)
