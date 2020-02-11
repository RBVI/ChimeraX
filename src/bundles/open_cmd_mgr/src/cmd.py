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

from chimerax.core.commands import CmdDesc, register, FileNameArg, RestOfLine, next_token
from chimerax.core.errors import UserError

def cmd_open(session, file_name, rest_of_line):
    from .manager import _manager as mgr
    tokens = []
    remainder = rest_of_line
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
    print("data_format:", data_format, " database name:", database_name)

    #return _cmd(session, test_atoms, name, hbond_allowance, overlap_cutoff, "clashes", color, radius, **kw)

def register_command(command_name, logger):
    register('open2', CmdDesc(required=[('file_name', FileNameArg), ('rest_of_line', RestOfLine)],
        synopsis="Open/fetch data files"), cmd_open, logger=logger)
