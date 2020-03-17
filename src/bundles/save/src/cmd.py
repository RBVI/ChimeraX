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
    mgr = session.save
    data_format= file_format(session, file_name, format_name)
    try:
        provider_args = mgr.save_args(data_format)
    except NoSaverError as e:
        raise LimitationError(str(e))

    provider_cmd_text = "save " + " ".join([FileNameArg.unparse(file_name)] + tokens)
    # register a private 'save' command that handles the provider's keywords
    registry = RegisteredCommandInfo()
    def format_names(formats=session.data_formats.formats):
        names = set()
        for f in formats:
            names.update(f.nicknames)
        return names

    keywords = {
        'format': DynamicEnum(format_names),
    }
    for keyword, annotation in provider_args.items():
        if keyword in keywords:
            raise ValueError("Save-provider keyword '%s' conflicts with builtin arg of same name" % keyword)
        keywords[keyword] = annotation
    desc = CmdDesc(required=[('file_name', SaveFileNameArg)], keyword=keywords.items(),
        synopsis="unnecessary")
    register("save", desc, provider_save, registry=registry)
    Command(session, registry=registry).run(provider_cmd_text, log=log)

def provider_save(session, file_name, format=None, **provider_kw):
    mgr = session.save
    data_format = file_format(session, file_name, format)
    bundle_info, provider_name = mgr.save_info(data_format)
    path = _get_path(file_name)
    return bundle_info.run_provider(session, provider_name, mgr).save(session, path, **provider_kw)

def _get_path(file_name):
    from os.path import expanduser, expandvars, exists
    return expanduser(expandvars(file_name))

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
            dot_pos = file_name.rindex('.')
            data_format = session.data_formats.data_format_from_suffix(file_name[dot_pos:])
            if not data_format:
                raise UserError("No known data format for file suffix '%s'" % file_name[dot_pos:])
        else:
            raise UserError("Cannot determine format for '%s'" % file_name)
    return data_format


def register_command(command_name, logger):
    register('save2', CmdDesc(
        required=[('file_name', SaveFileNameArg), ('rest_of_line', RestOfLine)],
        synopsis="Save file", self_logging=True), cmd_save, logger=logger)
