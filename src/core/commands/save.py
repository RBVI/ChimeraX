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

def save(session, filename, models=None, format=None, **kw):
    '''Save data, sessions, images.  Specific formats have additional keyword arguments using
    commands.add_keyword_arguments().  These are listed with command "usage save".

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
            fnames = sum([tuple(f.nicknames) for f in io.formats(open=False)], ())
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

    if models:
        kw["models"] = models
    try:
        fmt.export(session, filename, format, **kw)
    except ValueError:
        # No export function registered
        suffixes = ', '.join(sum([f.extensions for f in io.formats() if f.export_func], []))
        from ..errors import UserError
        if not suffix:
            msg = 'Missing file suffix, require one of %s' % suffixes
        else:
            msg = 'Unrecognized file suffix "%s", require one of %s' % (suffix, suffixes)
        raise UserError(msg)
    except TypeError as e:
        # Keywords incompatible with export function
        if 'unexpected keyword' in str(e):
            from ..errors import UserError
            raise UserError(str(e))
        raise

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
            from html import escape
            if f.reference:
                descrip = '<a href="%s">%s</a>' % (f.reference, escape(f.synopsis))
            else:
                descrip = escape(f.synopsis)
            lines.append('<tr><td>%s<td>%s<td>%s' % (descrip,
                escape(commas(f.nicknames)), escape(', '.join(f.extensions))))
        else:
            session.logger.info('    %s: %s: %s' % (f.synopsis,
                commas(f.nicknames), ', '.join(f.extensions)))
    if session.ui.is_gui:
        lines.append('</table>')
        msg = '\n'.join(lines)
        session.logger.info(msg, is_html=True)

from . import DynamicEnum
class SaveFileFormatsArg(DynamicEnum):
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
    from . import CmdDesc, register, SaveFileNameArg, ModelsArg
    desc = CmdDesc(
        required=[('filename', SaveFileNameArg)],
        optional=[('models', ModelsArg)],
        keyword=[('format', SaveFileFormatsArg())],
        synopsis='save data to various file formats'
    )
    register('save', desc, save, logger=session.logger)

    sf_desc = CmdDesc(synopsis='report formats that can be saved')
    register('save formats', sf_desc, save_formats, logger=session.logger)
