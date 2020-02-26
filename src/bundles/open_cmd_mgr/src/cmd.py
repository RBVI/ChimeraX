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
    file_name = file_names[0]
    if len(file_names) > 1:
        remainder = " ".join([FileNameArg.unparse(x) for x in file_names[1:]])
        if rest_of_line:
            remainder += " " + rest_of_line
    else:
        remainder = rest_of_line
    tokens = []
    while remainder:
        token, token_log, remainder = next_token(remainder)
        remainder = remainder.lstrip()
        tokens.append(token)
    database_name = format_name = None
    for i in range(len(tokens)-2, -1, -2):
        test_token = tokens[i].lower()
        if "format".startswith(test_token):
            format_name = tokens[i+1]
            break
        elif "fromdatabase".startswith(test_token):
            database_name = tokens[i+1]
            break
    data_format, database_name = process_file_info(session, file_name, format_name, database_name)

    from .manager import _manager as mgr, NoOpenerError
    if data_format:
        try:
            provider_args, want_path, check_path = mgr.open_info(data_format)
        except NoOpenerError as e:
            raise LimitationError(str(e))
    else:
        raise LimitationError("Revamped data fetching not yet implemented")
    provider_cmd_text = "open " + " ".join([FileNameArg.unparse(file_name)] + tokens)
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
    file_infos = [FileInfo(session, fn, format, from_database) for fn in file_names]
    formats = set([fi.data_format for fi in file_infos])
    databases = set([fi.database_name for fi in file_infos])
    if provider_kw and (len(formats) > 1 or len(databases) > 1):
        raise UserError("Cannot provide format/database-specific keywords when opening multiple different"
            " formats or databases; use several 'open' commands instead.")
    for fi in file_infos:
        if fi.database_name:
            #TODO: core.commands.open._fetch_from_database
            continue
        bundle_info, name, want_path, check_path = mgr.open_info(fi.data_format.name)

def process_file_info(session, file_name, format_name, database_name):
    if format_name:
        try:
            data_format = session.data_formats[format_name]
        except KeyError:
            raise UserError("Unknown data format: '%s'" % format_name)
    else:
        data_format = None
    # it's possible for a fetch to have no database specified and still have a format, e.g.
    # "open 1gcn format pdb"
    if not database_name:
        if ':' in file_name:
            database_name, file_name = file_name.split(':', 1)
            #TODO: when fetch actually implemented, check that it's a known fetch type and if not,
            # treat the whole thing as a file name
        elif not data_format:
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
            elif likely_pdb_id(file_name):
                database_name = "pdb"
            else:
                raise UserError("Cannot determine format for '%s'" % file_name)
        elif likely_pdb_id(file_name):
            # handle "open 1gcn format pdb"
            database_name = "pdb"
    return data_format, database_name

class FileInfo:
    def __init__(self, session, file_name, format_name, database_name):
        self.file_name = file_name
        self.data_format, self.database_name = process_file_info(
                    session, file_name, format_name, database_name)


def register_command(command_name, logger):
    register('open2', CmdDesc(
        required=[('file_names', OpenFileNamesArgNoRepeat), ('rest_of_line', RestOfLine)],
        synopsis="Open/fetch data files", self_logging=True), cmd_open, logger=logger)
