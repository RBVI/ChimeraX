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

def cmd_open(session, file_name, rest_of_line):
    print("file name:", repr(file_name), " rest of line:", repr(rest_of_line))
    #return _cmd(session, test_atoms, name, hbond_allowance, overlap_cutoff, "clashes", color, radius, **kw)

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, FileNameArg, RestOfLine
    register('open2', CmdDesc(required=[('file_name', FileNameArg), ('rest_of_line', RestOfLine)],
        synopsis="Open/fetch data files"), cmd_open, logger=logger)
