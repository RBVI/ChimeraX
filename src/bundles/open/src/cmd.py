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
    return not os.path.exists(text) and len(text) == 4 and text[0].isdigit() and text[1:].isalphanum()

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

    from .manager import _manager as mgr, NoOpenerError
    fetches, files = fetches_vs_files(file_names, format_name, database_name)
    if fetches:
        #TODO
        raise LimitationError("Revamped data fetching not yet implemented")
    else:
        data_format= file_format(session, files[0], format_name)
        try:
            provider_args, want_path, check_path, batch = mgr.open_info(data_format)
        except NoOpenerError as e:
            raise LimitationError(str(e))

    provider_cmd_text = "open " + " ".join([FileNameArg.unparse(fn) for fn in file_names] + tokens)
    # register a private 'open' command that handles the provider's keywords
    registry = RegisteredCommandInfo()
    def format_names(formats=session.data_formats.formats):
        names = set()
        for f in formats:
            names.update(f.nicknames)
        return names

    keywords = {
        'format': DynamicEnum(format_names),
        #TODO: keywords['from_database'] = DynamicEnum(database_names)
        'ignore_cache': BoolArg,
        'name': StringArg
    }
    for keyword, annotation in provider_args.items():
        if keyword in keywords:
            raise ValueError("Open-provider keyword '%s' conflicts with builtin arg of same name" % keyword)
        keywords[keyword] = annotation
    desc = CmdDesc(required=[('file_names', OpenFileNamesArg)], keyword=keywords.items(),
        synopsis="unnecessary")
    register("open", desc, provider_open, registry=registry)
    Command(session, registry=registry).run(provider_cmd_text, log=log)

def provider_open(session, file_names, format=None, from_database=None, ignore_cache=False, name=None,
        **provider_kw):
    # since the "file names" may be globs, need to preprocess them...
    fetches, file_names = fetches_vs_files(file_names, format, from_database)
    file_infos = [FileInfo(session, fn, format) for fn in file_names]
    formats = set([fi.data_format for fi in file_infos])
    databases = set([f[1:] for f in fetches])
    homogeneous = len(formats) +  len(databases) == 1
    if provider_kw and not homogeneous:
        raise UserError("Cannot provide format/database-specific keywords when opening multiple different"
            " formats or databases; use several 'open' commands instead.")
    opened_models = []
    if homogenous:
        #TODO: continue revamp
        data_format, database_name = formats.pop(), databases.pop()
        if database_name:
            #TODO: core.commands.open._fetch_from_database
            pass
        else:
            bundle_info, provider_name, want_path, check_path, batch = mgr.open_info(data_format.name)
            if batch:
                paths = [_get_path(fi.file_name, check_path) for fi in file_infos]
                models = bundle_info.run_provider(session, self, provider_name,
                                    operation="open", data=paths, **provider_kw)
                name_models(models, name, paths)
                opened_models.extend(models)
            else:
                for fi in file_infos:
                    if want_path:
                        data = _get_path(fi.file_name, check_path)
                    else:
                        data = _get_stream(fi.file_name, data_format.encoding)
                    models = bundle_info.run_provider(session, self, provider_name,
                                    operation="open", data=data, **provider_kw)
                    name_models(models, name, fi.file_name)
                    opened_models.extend(models)
    else:
        for fi in file_infos:
            if fi.database_name:
                #TODO: core.commands.open._fetch_from_database
                pass
            else:
                bundle_info, provider_name, want_path, check_path, batch = mgr.open_info(fi.data_format.name)
                for fi in file_infos:
                    if want_path:
                        data = _get_path(fi.file_name, check_path)
                    else:
                        data = _get_stream(fi.file_name, fi.data_format.encoding)
                    models = bundle_info.run_provider(session, self, provider_name,
                                    operation="open", data=data, **provider_kw)
                    name_models(models, name, fi.file_name)
                    opened_models.extend(models)
    if opened_models:
        session.models.add(opened_models)
    return opened_models

def _get_path(file_name, check_path, check_compression=True):
    from os.path import expanduser, expandvars, exists
    expanded = expanduser(expandvars(file_name))
    if check_path and not exists(expanded):
        raise UserError("No such file/path: %s" % file_name)

    if check_compression:
        from .manager import _manager as mgr
        if mgr.remove_compression_suffix(expanded) != expanded:
            raise UserError("File reader requires uncompressed file; '%s' is compressed" % file_name)
    return expanded

def _get_stream(file_name, encoding):
    path = _get_path(file_name, True, check_compression=False)
    from .manager import _manager as mgr
    return mgr.open_file(path, encoding)

def fetches_vs_files(names, format_name, database_name):
    fetches = []
    files = []
    from os.path import exists
    for name in names:
        if not database_name and exists(name):
            files.append(name)
        else:
            f = fetch_info(name, format_name, database_name)
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

def fetch_info(identifier, format_name, database_name):
    #TODO
    return None

def name_models(models, name_arg, path_info):
    if not models:
        return
    if name_arg:
        for m in models:
            m.name = name_arg
        return
    if isinstance(path_info, str):
        model_name = model_name_from_path(path_info)
    elif len(path_info) != len(models):
        model_name = model_name_from_path(path_info[0])
    else:
        for m, path in zip(models, path_info):
            if not m.name:
                m.name = model_name_from_path(path)
        return
    for m in models:
        if not m.name:
            m.name = model_name

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
            data_format = session.data_formats[format_name]
        except KeyError:
            raise UserError("Unknown data format: '%s'" % format_name)
    else:
        data_format = None

    if not data_format:
        if '.' in file_name:
            from .manager import _manager as mgr
            base_name = mgr.remove_compression_suffix(file_name)
            try:
                dot_pos = base_name.rindex('.')
            except ValueError:
                raise UserError("'%s' has only compression suffix; cannot determine format from suffix"
                    % file_name)
            data_format = session.data_formats.data_format_from_suffix(base_name[dot_pos:])
            if not data_format:
                raise UserError("No known data format for file suffix '%s'" % base_name[dot_pos:])
        else:
            raise UserError("Cannot determine format for '%s'" % file_name)
    return data_format

class FileInfo:
    def __init__(self, session, file_name, format_name):
        self.file_name = file_name
        self.data_format = process_file_info(session, file_name, format_name)


def register_command(command_name, logger):
    register('open2', CmdDesc(
        required=[('file_names', OpenFileNamesArgNoRepeat), ('rest_of_line', RestOfLine)],
        synopsis="Open/fetch data files", self_logging=True), cmd_open, logger=logger)
