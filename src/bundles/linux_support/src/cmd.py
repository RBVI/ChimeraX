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

import sys
from chimerax.core.commands import CmdDesc, BoolArg, StringArg
from chimerax.core.errors import UserError


def linux_xdg_install(session, verbose=False, system=False):
    if sys.platform != "linux":
        raise UserError("Only runs on Linux")
    from .xdg import install
    install(session, verbose=verbose, system=system)


linux_xdg_install_desc = CmdDesc(
    keyword=[
        ("verbose", BoolArg),
        ("system", BoolArg),
    ],
    hidden=["system"],
    synopsis='Install desktop menu and icons')


def linux_xdg_uninstall(session, verbose=False, system=False):
    if sys.platform != "linux":
        raise UserError("Only runs on Linux")
    from .xdg import uninstall
    uninstall(session, verbose=verbose, system=system)


linux_xdg_uninstall_desc = CmdDesc(
    keyword=[
        ("verbose", BoolArg),
        ("system", BoolArg),
    ],
    hidden=["system"],
    synopsis='Uninstall desktop menu and icons')


def linux_flatpak_files(session, ident, verbose=False):
    if sys.platform != "linux":
        raise UserError("Only runs on Linux")
    from .flatpak import flatpak_files
    flatpak_files(session, ident, verbose=verbose)


linux_flatpak_files_desc = CmdDesc(
    required=[
        ("ident", StringArg),
    ],
    keyword=[
        ("verbose", BoolArg),
    ],
    synopsis='Generate flatpak metainfo and desktop files')
