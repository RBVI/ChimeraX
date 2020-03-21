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

from chimerax.core.commands import CmdDesc, register, Command, OpenFileNamesArg, RestOfLine, next_token, \
    FileNameArg, BoolArg, StringArg, DynamicEnum
from chimerax.core.commands.cli import RegisteredCommandInfo
from chimerax.core.errors import UserError, LimitationError

# need to use non-repeatable OpenFilesNamesArg (rather than OpenFileNameArg) so that 'browse' can still be
# used to open multiple files
class OpenFileNamesArgNoRepeat(OpenFileNamesArg):
    allow_repeat = False

import os.path
def likely_pdb_id(text):
    return not os.path.exists(text) and len(text) == 4 and text[0].isdigit() and text[1:].isalnum()

def cmd_open(session, file_names, rest_of_line, *, log=True):
    tokens = []
    remainder = rest_of_line
    while remainder:
        token, token_log, remainder = next_token(remainder)
        remainder = remainder.lstrip()
        tokens.append(token)
    database_name = format_name = None
    for i in range(len(tokens)-2, -1, -2):
        test_token = tokens[i].lower()
        if "format".startswith(test_token):
            format_name = tokens[i+1]
        elif "fromdatabase".startswith(test_token):
            database_name = tokens[i+1]

    from .manager import NoOpenerError
    mgr = session.open_command
    fetches, files = fetches_vs_files(mgr, file_names, format_name, database_name)
    if fetches:
        try:
            provider_args = mgr.fetch_args(fetches[0][1], format_name=fetches[0][2])
        except NoOpenerError as e:
            raise LimitationError(str(e))
    else:
        data_format = file_format(session, files[0], format_name)
        try:
            provider_args = mgr.open_args(data_format)
        except NoOpenerError as e:
            raise LimitationError(str(e))

    provider_cmd_text = "open " + " ".join([FileNameArg.unparse(fn)
        for fn in file_names] + tokens)
    # register a private 'open' command that handles the provider's keywords
    registry = RegisteredCommandInfo()
    def format_names(formats=session.data_formats.formats):
        names = set()
        for f in formats:
            names.update(f.nicknames)
        return names

    def database_names(mgr=mgr):
        return mgr.database_names

    keywords = {
        'format': DynamicEnum(format_names),
        'from_database': DynamicEnum(database_names),
        'ignore_cache': BoolArg,
        'name': StringArg
    }
    for keyword, annotation in provider_args.items():
        if keyword in keywords:
            raise ValueError("Open-provider keyword '%s' conflicts with builtin arg of"
                " same name" % keyword)
        keywords[keyword] = annotation
    desc = CmdDesc(required=[('names', OpenFileNamesArg)], keyword=keywords.items(),
        synopsis="unnecessary")
    register("open", desc, provider_open, registry=registry)
    Command(session, registry=registry).run(provider_cmd_text, log=log)

def provider_open(session, names, format=None, from_database=None, ignore_cache=False, name=None,
        return_status=False, _add_to_file_history=True, **provider_kw):
    mgr = session.open_command
    # since the "file names" may be globs, need to preprocess them...
    fetches, file_names = fetches_vs_files(mgr, names, format, from_database)
    file_infos = [FileInfo(session, fn, format) for fn in file_names]
    formats = set([fi.data_format for fi in file_infos])
    databases = set([f[1:] for f in fetches])
    homogeneous = len(formats) +  len(databases) == 1
    if provider_kw and not homogeneous:
        raise UserError("Cannot provide format/database-specific keywords when opening"
            " multiple different formats or databases; use several 'open' commands"
            " instead.")
    opened_models = []
    statuses = []
    if homogeneous:
        data_format = formats.pop() if formats else None
        database_name, format = databases.pop() if databases else (None, format)
        if database_name:
            fetcher_info, default_format_name = _fetch_info(mgr, database_name, format)
            for ident, database_name, format_name in fetches:
                if format_name is None:
                    format_name = default_format_name
                models, status = fetcher_info.fetch(session, ident, format_name,
                    ignore_cache, **provider_kw)
                if status:
                    statuses.append(status)
                if models:
                    opened_models.append(name_and_group_models(models, name, [ident]))
        else:
            opener_info, provider_name, want_path, check_path, batch = mgr.open_info(
                data_format)
            if batch:
                paths = [_get_path(mgr, fi.file_name, check_path) for fi in file_infos]
                models, status = opener_info.open(session, paths, name, **provider_kw)
                if status:
                    statuses.append(status)
                if models:
                    opened_models.append(name_and_group_models(models, name, paths))
            else:
                for fi in file_infos:
                    if want_path:
                        data = _get_path(mgr, fi.file_name, check_path)
                    else:
                        data = _get_stream(mgr, fi.file_name, data_format.encoding)
                    models, status = opener_info.open(session, data,
                        name or model_name_from_path(fi.file_name), **provider_kw)
                    if status:
                        statuses.append(status)
                    if models:
                        opened_models.append(name_and_group_models(models, name,
                            [fi.file_name]))
    else:
        for fi in file_infos:
            opener_info, provider_name, want_path, check_path, batch = mgr.open_info(
                fi.data_format)
            for fi in file_infos:
                if want_path:
                    data = _get_path(mgr, fi.file_name, check_path)
                else:
                    data = _get_stream(mgr, fi.file_name, fi.data_format.encoding)
                models, status = opener_info.open(session, data,
                    name or model_name_from_path(fi.file_name), **provider_kw)
                if status:
                    statuses.append(status)
                if models:
                    opened_models.append(name_and_group_models(models, name, [fi.file_name]))
        for ident, database_name, format_name in fetches:
            fetcher_info, default_format_name = _fetch_info(mgr, database_name, format)
            if format_name is None:
                format_name = default_format_name
            models, status = fetcher_info.fetch(session, ident, format_name,
                ignore_cache, **provider_kw)
            if status:
                statuses.append(status)
            if models:
                opened_models.append(name_and_group_models(models, name, [ident]))
    if opened_models:
        session.models.add(opened_models)
    if _add_to_file_history and len(names) == 1:
        # TODO: Handle lists of file names in history
        from chimerax.core.filehistory import remember_file
        if fetches:
            # Files opened in the help browser are done asynchronously and might have
            # been misspelled and can't be deleted from file history.  So skip them.
            if not statuses or not statuses[-1].endswith(" in browser"):
                remember_file(session, names[0], format_name, opened_models or
                    'all models', database=database_name, open_options=provider_kw)
        else:
            remember_file(session, names[0], file_infos[0].data_format.name,
                opened_models or 'all models', open_options=provider_kw)

    status ='\n'.join(statuses) if statuses else ""
    if return_status:
        return opened_models, status
    elif status:
        session.logger.status(status, log=True)
    return opened_models

def _fetch_info(mgr, database_name, default_format_name):
    db_info = mgr.database_info(database_name)
    if default_format_name:
        try:
            bundle_info, is_default = db_info[default_format_name]
        except KeyError:
            raise UserError("Format '%s' not available for database '%s'.  Available"
                " formats are: %s" % (default_format_name, database_name,
                ", ".join(db_info.keys())))
    else:
        for default_format_name, fmt_info in db_info.items():
            bundle_info, is_default = fmt_info
            if is_default:
                break
        else:
            raise UserError("No default format for database '%s'.  Possible formats are:"
                " %s" % (database_name, ", ".join(db_info.keys())))
    return bundle_info.run_provider(mgr.session, database_name, mgr), default_format_name

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
    return io.open_input(path, encoding)

def fetches_vs_files(mgr, names, format_name, database_name):
    fetches = []
    files = []
    from os.path import exists
    for name in names:
        if not database_name and exists(name):
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
    from os.path import exists
    if not database_name and exists(file_arg):
        return None
    if ':' in file_arg:
        db_name, ident = file_arg.split(':', maxsplit=1)
    elif database_name:
        db_name = database_name
        ident = file_arg
    elif likely_pdb_id(file_arg):
        db_name = "pdb"
        ident = file_arg
    else:
        return None
    from .manager import NoOpenerError
    try:
        db_formats = mgr.database_info(db_name).keys()
    except NoOpenerError as e:
        raise LimitationError(str(e))
    if format_name and format_name not in [dbf for dbf in db_formats]:
        raise UserError("Format '%s' not supported for database '%s'.  Supported formats are: %s"
            % (format_name, db_name, ", ".join([dbf for dbf in db_formats])))
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

def file_format(session, file_name, format_name):
    if format_name:
        try:
            return session.data_formats[format_name]
        except KeyError:
            raise UserError("Unknown data format: '%s'" % format_name)

    return session.data_formats.file_name_to_format(file_name)

class FileInfo:
    def __init__(self, session, file_name, format_name):
        self.file_name = file_name
        self.data_format = file_format(session, file_name, format_name)


def register_command(command_name, logger):
    register('open2', CmdDesc(required=[('file_names', OpenFileNamesArgNoRepeat),
        ('rest_of_line', RestOfLine)], synopsis="Open/fetch data files",
        self_logging=True), cmd_open, logger=logger)
