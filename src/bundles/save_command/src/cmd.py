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

from chimerax.core.commands import CmdDesc, register, Command, SaveFileNameArg, RestOfLine, next_token, \
    FileNameArg, DynamicEnum, StringArg
from chimerax.core.commands.cli import RegisteredCommandInfo, log_command
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
    provider_cmd_text = "save " + " ".join([FileNameArg.unparse(file_name)]
        + [StringArg.unparse(token) for token in tokens])

    more_log_info = None
    try:
        from .manager import NoSaverError, SaverNotInstalledError
        mgr = session.save_command
        data_format = file_format(session, file_name, format_name, True, False)
        try:
            provider_args = mgr.save_args(data_format)
        except SaverNotInstalledError as e:
            from chimerax.core import toolshed
            bi = mgr.provider_info(data_format).bundle_info
            more_log_info = '<a href="%s">Install the %s bundle</a> to save "%s" format files.' % (
                toolshed.get_toolshed().bundle_url(bi.name), bi.short_name, data_format.name)
            raise LimitationError("%s; see log for more info" % e)
        except NoSaverError as e:
            raise LimitationError(str(e))

        # register a private 'save' command that handles the provider's keywords
        registry = RegisteredCommandInfo()
        keywords = {
            'format': DynamicEnum(lambda ses=session: format_names(ses)),
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
        desc = CmdDesc(required=[('file_name', SaveFileNameArg)], optional=optional,
            keyword=keywords.items(), hidden=mgr.hidden_args(data_format), synopsis="unnecessary")
        register("save", desc, provider_save, registry=registry)
    except BaseException as e:
        # want to log command even for keyboard interrupts
        log_command(session, "save", provider_cmd_text, url=_main_save_CmdDesc.url)
        if more_log_info:
            session.logger.info(more_log_info, is_html=True)
        raise
    Command(session, registry=registry).run(provider_cmd_text, log=log)

def provider_save(session, file_name, format=None, **provider_kw):
    mgr = session.save_command
    data_format = file_format(session, file_name, format, False, True)
    provider_info = mgr.provider_info(data_format)
    path = _get_path(file_name, provider_info.compression_okay)

    # TODO: The following line does a graphics update so that if the save command is
    # exporting data in a script (e.g. scene export) the graphics is up to date.  Does
    # not seem like the ideal solution to put this update here.
    if data_format.category == "Generic 3D objects":
        session.update_loop.update_graphics_now()
    try:
        saver_info = provider_info.bundle_info.run_provider(session, provider_info.format_name, mgr)
        saver_info.save(session, path, **provider_kw)
    except (IOError, PermissionError, OSError) as e:
        raise UserError("Cannot save '%s': %s" % (file_name, e))

    # remember in file history if appropriate
    try:
        session.open_command.opener_info(data_format)
    except Exception:
        pass
    else:
        from os.path import isfile
        if saver_info.in_file_history and isfile(path):
            from chimerax.core.filehistory import remember_file
            remember_file(session, path, data_format.nicknames[0],
                provider_kw.get('models', 'all models'), file_saved=True)

def format_names(session):
    names = set()
    for f in session.save_command.save_data_formats:
        names.update(f.nicknames)
    return names

def _get_path(file_name, compression_okay):
    from os.path import expanduser, expandvars, exists
    expanded = expanduser(expandvars(file_name))
    if not compression_okay:
        from chimerax import io
        if io.remove_compression_suffix(expanded) != expanded:
            raise UserError("File saver not capable of writing compressed output files;"
                " '%s' implies compression" % file_name)
    return expanded

def file_format(session, file_name, format_name, clear_before, clear_after):
    if format_name:
        try:
            return session.data_formats[format_name]
        except KeyError:
            raise UserError("Unknown data format: '%s'" % format_name)

    from chimerax.data_formats import NoFormatError
    try:
        return session.data_formats.save_format_from_file_name(file_name, clear_cache_before=clear_before,
            cache_user_responses=True, clear_cache_after=clear_after)
    except NoFormatError as e:
        raise UserError(str(e))

def cmd_save_formats(session):
    '''Report file formats and suffixes that the save command knows about.'''
    from chimerax.core.commands import commas
    all_formats = session.save_command.save_data_formats
    by_category = {}
    for fmt in all_formats:
        by_category.setdefault(fmt.category.title(), []).append(fmt)
    titles = list(by_category.keys())
    titles.sort()
    lines = []
    from chimerax.core import toolshed
    ts = toolshed.get_toolshed()
    for title in titles:
        formats = by_category[title]
        if session.ui.is_gui:
            lines.extend([
                '<table border=1 cellspacing=0 cellpadding=2>',
                '<tr><th colspan="3">%s' % title,
                '<tr><th>File format<th>Short name(s)<th>Suffixes'
            ])
        else:
            session.logger.info(title)
            session.logger.info('File format, Short name(s), Suffixes:')
        formats.sort(key = lambda f: f.name.lower())
        some_uninstalled = False
        for f in formats:
            bundle_info = session.save_command.provider_info(f).bundle_info
            if session.ui.is_gui:
                from html import escape
                if not bundle_info.installed:
                    some_uninstalled = True
                    descrip = '<a href="%s">%s</a><sup>*</sup>' % (ts.bundle_url(bundle_info.name),
                        escape(f.synopsis))
                elif f.reference_url:
                    descrip = '<a href="%s">%s</a>' % (f.reference_url, escape(f.synopsis))
                else:
                    descrip = escape(f.synopsis)
                lines.append('<tr><td>%s<td>%s<td>%s' % (descrip,
                    escape(commas(f.nicknames)), escape(', '.join(f.suffixes))))
            else:
                if not bundle_info.installed:
                    some_uninstalled = True
                    session.logger.info('    %s (not installed): %s: %s' % (f.synopsis,
                        commas(f.nicknames), ', '.join(f.suffixes)))
                else:
                    session.logger.info('    %s: %s: %s' % (f.synopsis,
                        commas(f.nicknames), ', '.join(f.suffixes)))
        if session.ui.is_gui:
            lines.append('</table>')
            if some_uninstalled:
                lines.append('<sup>*</sup>Not installed; click on link to install<br>')
            lines.append('<br>')
    if session.ui.is_gui:
        msg = '\n'.join(lines)
        session.logger.info(msg, is_html=True)

def _usage_setup(session):
    if session.ui.is_gui:
        get_name = lambda arg: arg.html_name()
        cmd_fmt = "<b>%s</b>"
        arg_fmt = "<i>%s</i>"
        end_of_main_syntax = "<br>\n&nbsp;&nbsp;&nbsp;&nbsp;&mdash; <i>%s</i>\n"
        arg_syntax_append = "<br>\n&nbsp;&nbsp;%s"
        arg_syntax_join = "<br>\n&nbsp;&nbsp;"
        kw_fmt = ' <nobr>[<b>%s</b> <i>%s</i>]</nobr>'
    else:
        get_name = lambda arg: arg.name
        cmd_fmt = "%s"
        arg_fmt = "%s"
        end_of_main_syntax = " -- %s"
        arg_syntax_append = "\n%s"
        arg_syntax_join = "\n"
        kw_fmt = ' [%s _%s_]'
    return get_name, cmd_fmt, arg_fmt, end_of_main_syntax, arg_syntax_append, arg_syntax_join, kw_fmt

from chimerax.core.commands.cli import user_kw

def cmd_usage_save(session):
    '''Report the generic syntax for the 'save' command'''

    arg_syntax = []
    get_name, cmd_fmt, arg_fmt, end_of_main_syntax, arg_syntax_append, arg_syntax_join, kw_fmt = \
        _usage_setup(session)
    syntax = cmd_fmt % "save"

    syntax += ' ' + arg_fmt % "name"
    arg_syntax.append("%s: %s" % (arg_fmt % "name", get_name(SaveFileNameArg)))
    for kw_name, arg in [('format', DynamicEnum(lambda ses=session: format_names(ses)))]:
        if isinstance(arg, type):
            # class, not instance
            syntax += kw_fmt % (kw_name, get_name(arg))
        else:
            syntax += kw_fmt % (kw_name, kw_name)
            arg_syntax.append("%s: %s" % (arg_fmt % kw_name, get_name(arg)))

    format_desc = "format-specific arguments"
    syntax += ' [%s]' % (arg_fmt % format_desc)
    arg_syntax.append("%s: %s" % (arg_fmt % format_desc, "format-specific arguments;"
        " to see their syntax use the '%s %s' command, where %s is as per the above"
        % (cmd_fmt % "usage save format", arg_fmt % "format", arg_fmt % "format")))

    syntax += end_of_main_syntax % "save data to various file formats"

    syntax += arg_syntax_append % arg_syntax_join.join(arg_syntax)

    session.logger.info(syntax, is_html=session.ui.is_gui)

def cmd_usage_save_format(session, format):
    '''Report the syntax for the 'save' command for a partilar format'''

    arg_syntax = []
    get_name, cmd_fmt, arg_fmt, end_of_main_syntax, arg_syntax_append, arg_syntax_join, kw_fmt = \
        _usage_setup(session)
    syntax = cmd_fmt % "save"

    syntax += ' ' + arg_fmt % "name"
    arg_syntax.append("%s: %s" % (arg_fmt % "names", get_name(SaveFileNameArg)))

    provider_args = session.save_command.save_args(session.data_formats[format])
    hidden_args = session.save_command.hidden_args(session.data_formats[format])
    for py_kw_name, arg in provider_args.items():
        if py_kw_name in hidden_args:
            continue
        kw_name = user_kw(py_kw_name)
        if isinstance(arg, type):
            # class, not instance
            syntax += kw_fmt % (kw_name, get_name(arg))
        else:
            syntax += kw_fmt % (kw_name, kw_name)
            arg_syntax.append("%s: %s" % (arg_fmt % kw_name, get_name(arg)))

    syntax += end_of_main_syntax % "save data to %s format" % format

    syntax += arg_syntax_append % arg_syntax_join.join(arg_syntax)

    session.logger.info(syntax, is_html=session.ui.is_gui)


_main_save_CmdDesc = None
def register_command(command_name, logger):
    global _main_save_CmdDesc
    _main_save_CmdDesc = CmdDesc(required=[('file_name', SaveFileNameArg), ('rest_of_line', RestOfLine)],
        synopsis="Save file", self_logging=True)
    register('save', _main_save_CmdDesc, cmd_save, logger=logger)

    sf_desc = CmdDesc(synopsis='report formats that can be saved')
    register('save formats', sf_desc, cmd_save_formats, logger=logger)

    us_desc = CmdDesc(synopsis='show generic "save" command syntax')
    register('usage save', us_desc, cmd_usage_save, logger=logger)

    usf_desc = CmdDesc(required=[('format', DynamicEnum(lambda ses=logger.session: format_names(ses)))],
        synopsis='show "save" command syntax for a specific file format')
    register('usage save format', usf_desc, cmd_usage_save_format, logger=logger)

