# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2020 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, EnumOf
    from . import tool

    DialogTypeArg = EnumOf(*zip(*((dt, dt.name.lower()) for dt in tool.DialogType)))

    toolshed_updates_desc = CmdDesc(
        optional=[('dialog_type', DialogTypeArg)],
        synopsis='show updates for installed bundles')

    def toolshed_updates(session, dialog_type=None):
        tool.show(session, dialog_type)

    register("toolshed updates", toolshed_updates_desc, toolshed_updates, logger=logger)
