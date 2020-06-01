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

from chimerax.core.commands import CmdDesc, register, Command, SaveFileNameArg, RestOfLine, next_token, \
    FileNameArg, DynamicEnum
from chimerax.core.commands.cli import RegisteredCommandInfo
from chimerax.core.errors import UserError, LimitationError

def cmd_save(session, file_name, rest_of_line, *, log=True):
    tokens = []
    remainder = rest_of_line
    while remainder:
        token, token_log, remainder = next_token(remainder)
        remainder = remainder.lstrip()
        tokens.append(token)
    format_name = None
    for i in range(len(tokens)-2, -1, -2):
        test_token = tokens[i].lower()
        if "format".startswith(test_token):
            format_name = tokens[i+1]

    from .manager import NoSaverError
    mgr = session.save_command
    data_format= file_format(session, file_name, format_name)
    try:
        provider_args = mgr.save_args(data_format)
    except NoSaverError as e:
        raise LimitationError(str(e))

    provider_cmd_text = "save " + " ".join([FileNameArg.unparse(file_name)] + tokens)
    # register a private 'save' command that handles the provider's keywords
    registry = RegisteredCommandInfo()
    def format_names(ses=session):
        names = set()
        for f in ses.save_command.save_data_formats:
            names.update(f.nicknames)
        return names

    keywords = {
        'format': DynamicEnum(format_names),
    }
    for keyword, annotation in provider_args.items():
        if keyword in keywords:
            raise ValueError("Save-provider keyword '%s' conflicts with builtin arg"
                " of same name" % keyword)
        keywords[keyword] = annotation
    # for convenience, allow 'models' to be a second positional argument instead of a keyword
    if 'models' in keywords:
        optional = [('models', keywords['models'])]
        del keywords['models']
    else:
        optional = []
    desc = CmdDesc(required=[('file_name', SaveFileNameArg)], optional=optional, keyword=keywords.items(),
        hidden=mgr.hidden_args(data_format), synopsis="unnecessary")
    register("save", desc, provider_save, registry=registry)
    Command(session, registry=registry).run(provider_cmd_text, log=log)

def provider_save(session, file_name, format=None, **provider_kw):
    mgr = session.save_command
    data_format = file_format(session, file_name, format)
    provider_info = mgr.save_info(data_format)
    path = _get_path(file_name, provider_info.compression_okay)

    # TODO: The following line does a graphics update so that if the save command is
    # exporting data in a script (e.g. scene export) the graphics is up to date.  Does
    # not seem like the ideal solution to put this update here.
    session.update_loop.update_graphics_now()
    provider_info.bundle_info.run_provider(session, provider_info.format_name,
        mgr).save(session, path, **provider_kw)

    # remember in file history if appropriate
    try:
        session.open_command.open_info(data_format)
    except Exception:
        pass
    else:
        from os.path import isfile
        if data_format.category != "Image" and isfile(path):
            from chimerax.core.filehistory import remember_file
            remember_file(session, path, data_format.nicknames[0],
                provider_kw.get('models', 'all models'), file_saved=True)

def _get_path(file_name, compression_okay):
    from os.path import expanduser, expandvars, exists
    expanded = expanduser(expandvars(file_name))
    if not compression_okay:
        from chimerax import io
        if io.remove_compression_suffix(expanded) != expanded:
            raise UserError("File saver requires uncompressed output file;"
                " '%s' would be compressed" % file_name)
    return expanded

def file_format(session, file_name, format_name):
    if format_name:
        try:
            return session.data_formats[format_name]
        except KeyError:
            raise UserError("Unknown data format: '%s'" % format_name)

    from chimerax.data_formats import NoFormatError
    try:
        return session.data_formats.save_format_from_file_name(file_name)
    except NoFormatError as e:
        raise UserError(str(e))

def cmd_save_formats(session):
    '''Report file formats and suffixes that the save command knows about.'''
    if session.ui.is_gui:
        lines = ['<table border=1 cellspacing=0 cellpadding=2>', '<tr><th>File format<th>Short name(s)<th>Suffixes']
    else:
        session.logger.info('File format, Short name(s), Suffixes:')
    from chimerax.core.commands import commas
    formats = session.save_command.save_data_formats
    formats.sort(key = lambda f: f.name)
    for f in formats:
        if session.ui.is_gui:
            from html import escape
            if f.reference_url:
                descrip = '<a href="%s">%s</a>' % (f.reference_url, escape(f.synopsis))
            else:
                descrip = escape(f.synopsis)
            lines.append('<tr><td>%s<td>%s<td>%s' % (descrip,
                escape(commas(f.nicknames)), escape(', '.join(f.suffixes))))
        else:
            session.logger.info('    %s: %s: %s' % (f.synopsis,
                commas(f.nicknames), ', '.join(f.suffixes)))
    if session.ui.is_gui:
        lines.append('</table>')
        msg = '\n'.join(lines)
        session.logger.info(msg, is_html=True)


def register_command(command_name, logger):
    register('save', CmdDesc(
        required=[('file_name', SaveFileNameArg), ('rest_of_line', RestOfLine)],
        synopsis="Save file", self_logging=True), cmd_save, logger=logger)

    sf_desc = CmdDesc(synopsis='report formats that can be saved')
    register('save formats', sf_desc, cmd_save_formats, logger=logger)
