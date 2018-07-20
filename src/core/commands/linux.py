# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2018 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.commands import CmdDesc, BoolArg


def install(session, system=False):
    from .._xdg import install
    install(session, system=system)

install_desc = CmdDesc(optional=[("system", BoolArg)],
                           synopsis='Install desktop menu and icons')


def uninstall(session, system=False):
    from .._xdg import uninstall
    uninstall(session, system=system)

uninstall_desc = CmdDesc(optional=[("system", BoolArg)],
                           synopsis='Uninstall desktop menu and icons')


def register_command(logger):
    from chimerax.core.commands import register

    register("linux xdg-install", install_desc, install,
             logger=logger)
    register("linux xdg-uninstall", uninstall_desc, uninstall,
             logger=logger)
