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

def runscript(session, text, *, log=True, downgrade_errors=False):
    """execute a Python script with arguments

    Parameters
    ----------
    text : string
        The text of the command to execute.
    log : bool
        Print the command text to the reply log.
    downgrade_errors : bool
        True if errors in the command should be logged as informational.
    """

    import shlex
    from ..scripting import open_python_script
    from ..errors import UserError
    argv = shlex.split(text)
    open_python_script(session, argv[0], argv[0], argv=argv)
    return []

def register_command(session):
    from . import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(required=[('text', StringArg)],
                   optional=[('log', BoolArg),
                             ('downgrade_errors', BoolArg),
                         ],
                   synopsis='run a Python script with arguments')
    register('runscript', desc, runscript, logger=session.logger)
