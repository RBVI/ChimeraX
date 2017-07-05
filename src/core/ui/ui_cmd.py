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

# -----------------------------------------------------------------------------
# Implementation of "ui" command.
#
def register_ui_command(logger):

    from ..commands import CmdDesc, register
    from ..commands import BoolArg, StringArg

    ui_desc = CmdDesc(
        required=[('do_start', BoolArg), ('tool_name', StringArg)],
        synopsis = 'control whether a tool is launched at ChimeraX startup')
    register('ui autostart', ui_desc, ui_autostart, logger=logger)

# -----------------------------------------------------------------------------
#
def ui_autostart(session, do_start, tool_name):
    '''
    Control whether a tool is launched at ChimeraX startup
    '''

    settings = session.ui.settings
    autostart = settings.autostart[:]
    if do_start:
        if tool_name not in autostart:
            autostart.append(tool_name)
    else:
        while tool_name in autostart:
            autostart.remove(tool_name)
    settings.autostart = autostart
    settings.save()
