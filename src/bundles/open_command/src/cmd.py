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

from chimerax.core.commands import CmdDesc, register, Command, OpenFileNamesArg, RestOfLine, next_token, \
    FileNameArg, BoolArg, StringArg, DynamicEnum, ModelIdArg
from chimerax.core.commands.cli import RegisteredCommandInfo, log_command
from chimerax.core.errors import UserError, LimitationError

# need to use non-repeatable OpenFilesNamesArg (rather than OpenFileNameArg) so that 'browse' can still be
# used to open multiple files, and turn off checking for existence so that things like pdb:1gcn are okay
class OpenInputArgNoRepeat(OpenFileNamesArg):
    allow_repeat = False
    check_existence = False

class OpenInputArg(OpenFileNamesArg):
    check_existence = False

import os.path
def likely_pdb_id(text, format_name):
    return not exists_locally(text, format_name) \
        and ((len(text) == 4 and text[0].isdigit() and text[1:].isalnum())
            or (len(text) == 8 and text[:5].isdigit() and text[5:].isalnum()))

def exists_locally(text, format):
    # does that name exist on the file system, and if it does but has no suffix, is there a format?
    # [trying to avoid directories named for PDB ID codes from fouling up "open pdb-code"]
    import os.path
    if not os.path.exists(text):
        return False
    if '.' in text or format:
        return True
    return False

def cmd_open(session, file_names, rest_of_line, *, log=True, return_json=False):
    """If return_json is True, the returned JSON object has one name/value pair:
        (name) model specs
        (value) a list of atom specifiers, one for each model opened by the command
    """
    tokens = []
    remainder = rest_of_line
    while remainder:
        token, token_log, remainder = next_token(remainder)
        remainder = remainder.lstrip()
        tokens.append(token)
    provider_cmd_text = "open " + " ".join([FileNameArg.unparse(fn)
        for fn in file_names] + [StringArg.unparse(token) for token in tokens])
    more_log_info = None
    try:
        database_name = format_name = None
        for i in range(len(tokens)-2, -1, -2):
            test_token = tokens[i].lower()
            if "format".startswith(test_token):
                format_name = tokens[i+1]
            elif "fromdatabase".startswith(test_token):
                database_name = tokens[i+1]

        from .manager import NoOpenerError, OpenerNotInstalledError
        mgr = session.open_command
        fetches, files = fetches_vs_files(mgr, file_names, format_name, database_name)
        if fetches:
            try:
                provider_args = mgr.fetch_args(fetches[0][1], format_name=fetches[0][2])
            except NoOpenerError as e:
                raise LimitationError(str(e))
        else:
            data_format = file_format(session, files[0], format_name, True, False)
            if data_format is None:
                # let provider_open raise the error, which will show the command
                provider_args = {}
            else:
                try:
                    provider_args = mgr.open_args(data_format)
                except OpenerNotInstalledError as e:
                    from chimerax.core import toolshed
                    bi = mgr.provider_info(data_format).bundle_info
                    more_log_info = '<a href="%s">Install the %s bundle</a> to open "%s" format files.' % (
                        toolshed.get_toolshed().bundle_url(bi.name), bi.short_name, data_format.name)
                    raise LimitationError("%s; see log for more info" % e)
                except NoOpenerError as e:
                    raise LimitationError(str(e))

        # register a private 'open' command that handles the provider's keywords
        registry = RegisteredCommandInfo()

        def database_names(mgr=mgr):
            return mgr.database_names

        keywords = {
            'format': DynamicEnum(lambda ses=session:format_names(ses)),
            'from_database': DynamicEnum(database_names),
            'id': ModelIdArg,
            'ignore_cache': BoolArg,
            'name': StringArg
        }
        for keyword, annotation in provider_args.items():
            if keyword in keywords:
                raise ValueError("Open-provider keyword '%s' conflicts with builtin arg of"
                    " same name" % keyword)
            keywords[keyword] = annotation
        desc = CmdDesc(required=[('names', OpenInputArg)], keyword=keywords.items(),
            synopsis="read and display data")
        register("open", desc, provider_open, registry=registry)
    except BaseException as e:
        # want to log command even for keyboard interrupts
        log_command(session, "open", provider_cmd_text, url=_main_open_CmdDesc.url)
        if more_log_info:
            session.logger.info(more_log_info, is_html=True)
        raise
    # Unlike run(), Command.run returns a list of results
    models = Command(session, registry=registry).run(provider_cmd_text, log=log)[0]
    if return_json:
        from chimerax.core.commands import JSONResult
        from json import JSONEncoder
        open_data = { 'model specs': [m.string(style="command") for m in models] }
        return JSONResult(JSONEncoder().encode(open_data), models)
    return models

def provider_open(session, names, format=None, from_database=None, ignore_cache=False, name=None, id=None,
        _return_status=False, _add_models=True, _request_file_history=False, log_errors=True, **provider_kw):
    mgr = session.open_command
    # since the "file names" may be globs, need to preprocess them...
    fetches, file_names = fetches_vs_files(mgr, names, format, from_database)
    file_infos = [FileInfo(session, fn, format, False, fn is file_names[-1]) for fn in file_names]
    formats = set([fi.data_format for fi in file_infos])
    databases = set([f[1:] for f in fetches])
    homogeneous = len(formats) +  len(databases) == 1
    if provider_kw and not homogeneous:
        raise UserError("Cannot provide format/database-specific keywords when opening"
            " multiple different formats or databases; use several 'open' commands"
            " instead.")
    opened_models = []
    ungrouped_models = []
    statuses = []
    from chimerax.atomic import Structure
    if homogeneous:
        data_format = formats.pop() if formats else None
        database_name, format = databases.pop() if databases else (None, format)
        if database_name:
            fetcher_info, default_format_name, pregrouped_structures, group_multiple_models = _fetch_info(
                mgr, database_name, format)
            in_file_history = fetcher_info.in_file_history
            for ident, database_name, format_name in fetches:
                if format_name is None:
                    format_name = default_format_name
                models, status = collated_open(session, database_name, ident,
                    session.data_formats[format_name], _add_models, log_errors, fetcher_info.fetch,
                    (session, ident, format_name, ignore_cache), provider_kw)
                if status:
                    statuses.append(status)
                if models:
                    if group_multiple_models:
                        opened_models.append(name_and_group_models(models, name, [ident]))
                    else:
                        opened_models.extend(models)
                    if pregrouped_structures:
                        for model in models:
                            ungrouped_models.extend([m for m in model.all_models()
                                if isinstance(m, Structure)])
                    else:
                        ungrouped_models.extend(models)
        else:
            opener_info = mgr.opener_info(data_format)
            if opener_info is None:
                raise NotImplementedError("Don't know how to open uninstalled format %s" % data_format.name)
            in_file_history = opener_info.in_file_history
            provider_info = mgr.provider_info(data_format)
            if provider_info.batch:
                paths = [_get_path(mgr, fi.file_name, provider_info.check_path)
                    for fi in file_infos]
                models, status = collated_open(session, None, paths, data_format, _add_models, log_errors,
                opener_info.open, (session, paths, name), provider_kw)
                if status:
                    statuses.append(status)
                if models:
                    if provider_info.group_multiple_models:
                        opened_models.append(name_and_group_models(models, name, paths))
                    else:
                        opened_models.extend(models)
                    if provider_info.pregrouped_structures:
                        for model in models:
                            ungrouped_models.extend([m for m in model.all_models()
                                if isinstance(m, Structure)])
                    else:
                        ungrouped_models.extend(models)
            else:
                for fi in file_infos:
                    if provider_info.want_path:
                        data = _get_path(mgr, fi.file_name, provider_info.check_path)
                    else:
                        data = _get_stream(mgr, fi.file_name, data_format.encoding)
                    try:
                        models, status = collated_open(session, None, [data], data_format, _add_models,
                            log_errors, opener_info.open, (session, data,
                            name or model_name_from_path(fi.file_name)), provider_kw)
                    except UnicodeDecodeError:
                        if not provider_info.want_path and data_format.encoding == "utf-8":
                            # try utf-16/32 (see #8746)
                            for encoding in ['utf-16', 'utf-32']:
                                data.close()
                                try:
                                    data = _get_stream(mgr, fi.file_name, encoding)
                                    models, status = collated_open(session, None, [data], data_format,
                                        _add_models, log_errors, opener_info.open, (session, data,
                                        name or model_name_from_path(fi.file_name)), provider_kw)
                                except UnicodeDecodeError:
                                    continue
                                break
                            else:
                                raise
                        else:
                            raise
                    if status:
                        statuses.append(status)
                    if models:
                        if provider_info.group_multiple_models:
                            opened_models.append(name_and_group_models(models, name, [fi.file_name]))
                        else:
                            opened_models.extend(models)
                        if provider_info.pregrouped_structures:
                            for model in models:
                                ungrouped_models.extend([m for m in model.all_models()
                                    if isinstance(m, Structure)])
                        else:
                            ungrouped_models.extend(models)
    else:
        for fi in file_infos:
            opener_info = mgr.opener_info(fi.data_format)
            if opener_info is None:
                raise NotImplementedError("Don't know how to fetch uninstalled format %s"
                    % fi.data_format.name)
            in_file_history = opener_info.in_file_history
            provider_info = mgr.provider_info(fi.data_format)
            if provider_info.want_path:
                data = _get_path(mgr, fi.file_name, provider_info.check_path)
            else:
                data = _get_stream(mgr, fi.file_name, fi.data_format.encoding)
            models, status = collated_open(session, None, [data], fi.data_format, _add_models, log_errors,
                opener_info.open, (session, data, name or model_name_from_path(fi.file_name)), provider_kw)
            if status:
                statuses.append(status)
            if models:
                if provider_info.group_multiple_models:
                    opened_models.append(name_and_group_models(models, name, [fi.file_name]))
                else:
                    opened_models.extend(models)
                if provider_info.pregrouped_structures:
                    for model in models:
                        ungrouped_models.extend([m for m in model.all_models() if isinstance(m, Structure)])
                else:
                    ungrouped_models.extend(models)
        for ident, database_name, format_name in fetches:
            fetcher_info, default_format_name, pregrouped_structures, group_multiple_models = _fetch_info(
                mgr, database_name, format)
            in_file_history = fetcher_info.in_file_history
            if format_name is None:
                format_name = default_format_name
            models, status = collated_open(session, database_name, ident, session.data_formats[format_name],
                _add_models, log_errors, fetcher_info.fetch, (session, ident, format_name, ignore_cache),
                provider_kw)
            if status:
                statuses.append(status)
            if models:
                if group_multiple_models:
                    opened_models.append(name_and_group_models(models, name, [ident]))
                else:
                    opened_models.extend(models)
                if pregrouped_structures:
                    for model in models:
                        ungrouped_models.extend([m for m in model.all_models() if isinstance(m, Structure)])
                else:
                    ungrouped_models.extend(models)
    if opened_models and _add_models:
        session.models.add(opened_models)
        if id is not None:
            from chimerax.std_commands.rename import rename
            rename(session, opened_models, id=id)
    if (_add_models or _request_file_history) and len(names) == 1 and in_file_history:
        # TODO: Handle lists of file names in history
        from chimerax.core.filehistory import remember_file
        if fetches:
            remember_file(session, names[0], session.data_formats[format_name].nicknames[0],
                opened_models or 'all models', database=database_name, open_options=provider_kw)
        else:
            remember_file(session, names[0], file_infos[0].data_format.nicknames[0],
                opened_models or 'all models', open_options=provider_kw)

    status ='\n'.join(statuses) if statuses else ""
    if _return_status:
        return ungrouped_models, status
    else:
        session.logger.status(status, log=status)
    return ungrouped_models

def _fetch_info(mgr, database_name, default_format_name):
    db_info = mgr.database_info(database_name)
    from chimerax.core.commands import commas
    if default_format_name:
        try:
            provider_info = db_info[default_format_name]
        except KeyError:
            raise UserError("Format '%s' not available for database '%s'.  Available"
                " formats are: %s" % (default_format_name, database_name,
                commas(db_info.keys())))
    else:
        for default_format_name, provider_info in db_info.items():
            if provider_info.is_default:
                break
        else:
            raise UserError("No default format for database '%s'.  Possible formats are:"
                " %s" % (database_name, commas(db_info.keys())))
    return (provider_info.bundle_info.run_provider(mgr.session, database_name, mgr),
        default_format_name, provider_info.pregrouped_structures, provider_info.group_multiple_models)

def _get_path(mgr, file_name, check_path, check_compression=True):
    from os.path import expanduser, expandvars, exists
    expanded = expanduser(expandvars(file_name))
    from chimerax.io import file_system_file_name
    if check_path and not exists(file_system_file_name(expanded)):
        raise UserError("No such file/path: %s" % file_name)

    if check_compression:
        from chimerax import io
        if io.remove_compression_suffix(expanded) != expanded:
            raise UserError("File reader requires uncompressed file; '%s' is compressed"
                % file_name)
    return expanded

def _get_stream(mgr, file_name, encoding):
    path = _get_path(mgr, file_name, True, check_compression=False)
    from chimerax import io
    try:
        return io.open_input(path, encoding)
    except IsADirectoryError:
        raise UserError("'%s' is a folder, not a file" % path)
    except (IOError, PermissionError) as e:
        raise UserError("Cannot open '%s': %s" % (path, e))

def fetches_vs_files(mgr, names, format_name, database_name):
    fetches = []
    files = []
    for name in names:
        if not database_name and exists_locally(name, format_name):
            files.append(name)
        else:
            f = fetch_info(mgr, name, format_name, database_name)
            if f:
                fetches.append(f)
            else:
                files.extend(expand_path(name))
    return fetches, files

def expand_path(file_name):
    from os.path import exists
    if exists(file_name):
        return [file_name]

    from glob import glob
    file_names = glob(file_name)
    if not file_names:
        return [file_name]
    # python glob does not sort.  Keep series in order
    file_names.sort()
    return file_names

def fetch_info(mgr, file_arg, format_name, database_name):
    if not database_name and exists_locally(file_arg, format_name):
        return None
    if ':' in file_arg:
        db_name, ident = file_arg.split(':', maxsplit=1)
        if len(db_name) < 2:
            return None
    elif database_name:
        db_name = database_name
        ident = file_arg
    elif likely_pdb_id(file_arg, format_name):
        db_name = "pdb"
        ident = file_arg
    else:
        return None
    db_name = db_name.lower()
    from .manager import NoOpenerError
    try:
        db_formats = list(mgr.database_info(db_name).keys())
    except NoOpenerError as e:
        raise LimitationError(str(e))
    if format_name and format_name not in db_formats:
        # for backwards compatibiity, accept formal format name or nicknames
        try:
            df = mgr.session.data_formats[format_name]
        except KeyError:
            nicks = []
        else:
            nicks = df.nicknames + [df.name]
        for nick in nicks:
            if nick in db_formats:
                format_name = nick
                break
        else:
            from chimerax.core.commands import commas
            raise UserError("Format '%s' not supported for database '%s'.  Supported"
                " formats are: %s" % (format_name, db_name,
                commas([dbf for dbf in db_formats])))
    return (ident, db_name, format_name)

def name_and_group_models(models, name_arg, path_info):
    if len(models) > 1:
        # name arg only applies to group, not underlings
        if name_arg:
            names = [name_arg] * len(models)
        elif len(path_info) == len(models):
            names = [model_name_from_path(p) for p in path_info]
        else:
            names = [model_name_from_path(path_info[0])] * len(models)
        for m, pn in zip(models, names):
            if name_arg or not m.name:
                m.name = pn
        from chimerax.core.models import Model
        names = set([m.name for m in models])
        if len(names) == 1:
            group_name = names.pop() + " group"
        elif len(path_info) == 1:
            group_name = model_name_from_path(path_info[0])
        else:
            group_name = "group"
        group = Model(group_name, models[0].session)
        group.add(models)
        return group
    model = models[0]
    if name_arg:
        model.name = name_arg
    else:
        if not model.name:
            model.name = model_name_from_path(path_info[0])
    return model

def model_name_from_path(path):
    from os.path import basename, dirname
    name = basename(path)
    if name.strip() == '':
        # Path is a directory with trailing '/'.  Use directory name.
        name = basename(dirname(path))
    return name

def file_format(session, file_name, format_name, clear_before, clear_after):
    if format_name:
        try:
            return session.data_formats[format_name]
        except KeyError:
            return None

    from chimerax.data_formats import NoFormatError
    try:
        return session.data_formats.open_format_from_file_name(file_name, clear_cache_before=clear_before,
            cache_user_responses=True, clear_cache_after=clear_after)
    except NoFormatError as e:
        return None

def collated_open(session, database_name, data, data_format, main_opener, log_errors,
        func, func_args, func_kw):
    def remember_data_format(func=func, data_format=data_format, func_args=func_args, func_kw=func_kw,
            data=data):
        try:
            models, status = func(*func_args, **func_kw)
        except (IOError, PermissionError) as e:
            if isinstance(data, str):
                raise UserError("Cannot open '%s': %s" % (data, e))
            else:
                raise UserError("Cannot open files: %s" % e)
        for m in models:
            m.opened_data_format = data_format
        return models, status
    is_script = data_format.category == session.data_formats.CAT_SCRIPT
    if is_script:
        with session.in_script:
            return remember_data_format()
    from chimerax.core.logger import Collator
    if database_name:
        description = "Summary of feedback from opening %s fetched from %s" % (data, database_name)
    else:
        if len(data) > 1:
            opened_text = "files"
        else:
            if isinstance(data[0], str):
                opened_text = data[0]
            elif hasattr(data[0], 'name'):
                opened_text = data[0].name
            else:
                opened_text = "input"
        description = "Summary of feedback from opening %s" % opened_text
    if main_opener and data_format.category != session.data_formats.CAT_SESSION:
        with Collator(session.logger, description, log_errors):
            return remember_data_format()
    return remember_data_format()

class FileInfo:
    def __init__(self, session, file_name, format_name, clear_before, clear_after):
        self.file_name = file_name
        self.data_format = file_format(session, file_name, format_name, clear_before, clear_after)
        if self.data_format is None:
            from os.path import splitext
            from chimerax import io
            ext = splitext(io.remove_compression_suffix(file_name))[1]
            if ext:
                raise UserError("Unrecognized file suffix '%s'" % ext)
            raise UserError("'%s' has no suffix" % file_name)

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

def cmd_usage_open(session):
    '''Report the generic syntax for the 'open' command'''

    arg_syntax = []
    get_name, cmd_fmt, arg_fmt, end_of_main_syntax, arg_syntax_append, arg_syntax_join, kw_fmt = \
        _usage_setup(session)
    syntax = cmd_fmt % "open"

    syntax += ' ' + arg_fmt % "names"
    arg_syntax.append("%s: %s" % (arg_fmt % "names", get_name(OpenFileNamesArg)))
    for kw_name, arg in [('format', DynamicEnum(lambda ses=session: format_names(ses))),
            ('fromDatabase', DynamicEnum(lambda ses=session: ses.open_command.database_names)),
            ('name', StringArg)]:
        if isinstance(arg, type):
            # class, not instance
            syntax += kw_fmt % (kw_name, get_name(arg))
        else:
            syntax += kw_fmt % (kw_name, kw_name)
            arg_syntax.append("%s: %s" % (arg_fmt % kw_name, get_name(arg)))

    format_desc = "format/database-specific arguments"
    syntax += ' [%s]' % (arg_fmt % format_desc)
    arg_syntax.append("%s: %s" % (arg_fmt % format_desc, "format- or database-specific arguments;"
        " to see their syntax use '%s %s' or '%s %s' commands respectively, where %s and %s are as per"
        " the above" % (cmd_fmt % "usage open format", arg_fmt % "format", cmd_fmt % "usage open database",
        arg_fmt % "database", arg_fmt % "format", arg_fmt % "database")))

    syntax += end_of_main_syntax % "read and display data"

    syntax += arg_syntax_append % arg_syntax_join.join(arg_syntax)

    session.logger.info(syntax, is_html=session.ui.is_gui)

def cmd_usage_open_format(session, format):
    '''Report the syntax for the 'open' command for a partilar format'''

    arg_syntax = []
    get_name, cmd_fmt, arg_fmt, end_of_main_syntax, arg_syntax_append, arg_syntax_join, kw_fmt = \
        _usage_setup(session)
    syntax = cmd_fmt % "open"

    syntax += ' ' + arg_fmt % "names"
    arg_syntax.append("%s: %s" % (arg_fmt % "names", get_name(OpenFileNamesArg)))

    from .manager import NoOpenerError
    try:
        provider_args = session.open_command.open_args(session.data_formats[format])
    except NoOpenerError:
        raise UserError("'%s' is a database format type; use the command 'usage open database' with the"
            " corresponding database type instead" % format)
    for py_kw_name, arg in provider_args.items():
        kw_name = user_kw(py_kw_name)
        if isinstance(arg, type):
            # class, not instance
            syntax += kw_fmt % (kw_name, get_name(arg))
        else:
            syntax += kw_fmt % (kw_name, kw_name)
            arg_syntax.append("%s: %s" % (arg_fmt % kw_name, get_name(arg)))

    syntax += end_of_main_syntax % "read and display data"

    syntax += arg_syntax_append % arg_syntax_join.join(arg_syntax)

    session.logger.info(syntax, is_html=session.ui.is_gui)

def cmd_usage_open_database(session, database):
    '''Report the syntax for the 'open' command for a partilar database fetch'''

    arg_syntax = []
    get_name, cmd_fmt, arg_fmt, end_of_main_syntax, arg_syntax_append, arg_syntax_join, kw_fmt = \
        _usage_setup(session)
    syntax = cmd_fmt % "open"

    syntax += ' ' + arg_fmt % "names"
    arg_syntax.append("%s: %s" % (arg_fmt % "names", get_name(OpenFileNamesArg)))

    mgr = session.open_command
    args = { 'ignoreCache': BoolArg }
    args.update(mgr.fetch_args(database))
    defaults = [format_name for format_name, provider_info in mgr.database_info(database).items()
        if provider_info.is_default]
    if len(defaults) == 1:
        from .manager import NoOpenerError
        try:
            args.update(session.open_command.open_args(session.data_formats[defaults[0]]))
        except NoOpenerError:
            pass

    for py_kw_name, arg in args.items():
        kw_name = user_kw(py_kw_name)
        if isinstance(arg, type):
            # class, not instance
            syntax += kw_fmt % (kw_name, get_name(arg))
        else:
            syntax += kw_fmt % (kw_name, kw_name)
            arg_syntax.append("%s: %s" % (arg_fmt % kw_name, get_name(arg)))

    syntax += end_of_main_syntax % "read and display data"

    syntax += arg_syntax_append % arg_syntax_join.join(arg_syntax)

    session.logger.info(syntax, is_html=session.ui.is_gui)

def cmd_open_formats(session):
    '''Report file formats, suffixes and databases that the open command knows about.'''
    from chimerax.core.commands import commas
    all_formats = session.open_command.open_data_formats
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
        formats.sort(key = lambda f: f.synopsis.lower())
        some_uninstalled = False
        for f in formats:
            bundle_info = session.open_command.provider_info(f).bundle_info
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
        else:
            session.logger.info('\n')

    if session.ui.is_gui:
        lines.extend(['<table border=1 cellspacing=0 cellpadding=2>', '<tr><th>Database<th>Formats'])
    else:
        session.logger.info('Database, Formats:')
    database_names = session.open_command.database_names
    database_names.sort(key=lambda dbn: dbn.lower())
    for db_name in database_names:
        db_info = session.open_command.database_info(db_name)
        if 'web fetch' in db_info.keys() or db_name == 'help':
            continue
        for fmt_name, fetcher_info in db_info.items():
            if fetcher_info.is_default:
                default_name = session.data_formats[fmt_name].nicknames[0]
                break
        else:
            continue
        format_names = [session.data_formats[fmt_name].nicknames[0] for fmt_name in db_info.keys()]
        format_names.sort()
        format_names.remove(default_name)
        format_names.insert(0, default_name)
        if not session.ui.is_gui:
            session.logger.info('    %s: %s' % (db_name, ', '.join(format_names)))
            continue
        line = '<tr><td>%s<td>%s' % (db_name, ', '.join(format_names))
        lines.append(line)

    if session.ui.is_gui:
        lines.append('</table>')
        msg = '\n'.join(lines)
        session.logger.info(msg, is_html=True)

def format_names(session):
    fmt_names = set([ nick for fmt in session.open_command.open_data_formats for nick in fmt.nicknames ])
    for db_name in session.open_command.database_names:
        for fmt_name in session.open_command.database_info(db_name).keys():
            for nick in session.data_formats[fmt_name].nicknames:
                fmt_names.add(nick)
    return fmt_names

_main_open_CmdDesc = None
def register_command(command_name, logger):
    global _main_open_CmdDesc
    _main_open_CmdDesc = CmdDesc(required=[('file_names', OpenInputArgNoRepeat),
        ('rest_of_line', RestOfLine)], synopsis="Open/fetch data files", self_logging=True)
    register('open', _main_open_CmdDesc, cmd_open, logger=logger)

    uo_desc = CmdDesc(synopsis='show generic "open" command syntax')
    register('usage open', uo_desc, cmd_usage_open, logger=logger)

    uof_desc = CmdDesc(required=[('format', DynamicEnum(lambda ses=logger.session: format_names(ses)))],
        synopsis='show "open" command syntax for a specific file format')
    register('usage open format', uof_desc, cmd_usage_open_format, logger=logger)

    uod_desc = CmdDesc(
        required=[('database', DynamicEnum(lambda ses=logger.session: ses.open_command.database_names))],
        synopsis='show "open" command syntax for a specific database fetch')
    register('usage open database', uod_desc, cmd_usage_open_database, logger=logger)

    of_desc = CmdDesc(synopsis='report formats that can be opened')
    register('open formats', of_desc, cmd_open_formats, logger=logger)
