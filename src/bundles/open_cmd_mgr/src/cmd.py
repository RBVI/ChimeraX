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

def cmd_open(session, file_names, rest_of_line):
    from .manager import _manager as mgr, NoOpenerError
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
    database_name = data_format = None
    for i in range(len(tokens)-2, -1, -2):
        test_token = tokens[i].lower()
        if "format".startswith(test_token):
            format_name = tokens[i+1]
            try:
                data_format = session.data_formats[format_name]
            except KeyError:
                raise UserError("Unknown data format: '%s'" % format_name)
            break
        elif "fromdatabase".startswith(test_token):
            database_name = tokens[i+1]
            break
    if not data_format and not database_name:
        if ':' in file_name:
            database_name, file_name = file_name.split(':', 1)
            #TODO: when fetch actually implemented, check that it's a known fetch type and if not,
            # treat the whole thing as a file name
        elif '.' in file_name:
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
            database_name = "pdb"
    if data_format:
        try:
            provider_args, want_path, check_path = mgr.open_info(data_format)
        except NoOpenerError as e:
            raise LimitationError(str(e))
    else:
        raise LimitationError("Revamped data fetching not yet implemented")
    provider_cmd_text = "open " + " ".join([FileNameArg.unparse(file_name)] + tokens)
    print("provider-open command: %s" % repr(provider_cmd_text))
    # register a private 'open' command that handles the provider's keywords
    registry = RegisteredCommandInfo()
    def format_names(formats=session.data_formats.formats):
        names = set()
        for f in formats:
            names.update(f.nicknames)
        #TODO: fetch databases
        return names

    keywords = {
        'format': DynamicEnum(format_names),
        #'from_database':
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
    Command(session, registry=registry).run(provider_cmd_text, log=True)

def provider_open(session, file_names, format=None, ignore_cache=False, name=None, **provider_kw):
    print("provider open, file names:", file_names, " format:", format, " ignore cache:", ignore_cache, " name:", name, " provider kw:", provider_kw)


def register_command(command_name, logger):
    register('open2', CmdDesc(
        required=[('file_names', OpenFileNamesArgNoRepeat), ('rest_of_line', RestOfLine)],
        synopsis="Open/fetch data files"), cmd_open, logger=logger)
