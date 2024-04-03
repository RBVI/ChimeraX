# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===


def _file_open(session):
    from chimerax.open_command import show_open_file_dialog
    show_open_file_dialog(session)


def _file_recent(session):
    mw = session.ui.main_window
    mw.rapid_access_shown = not mw.rapid_access_shown


def _file_save(session):
    from chimerax.save_command import show_save_file_dialog
    show_save_file_dialog(session)


_providers = {
    "Open": _file_open,
    "Recent": _file_recent,
    "Save": _file_save,
    "Close": "close session",
    "Exit": "exit",
    "Undo": "undo",
    "Redo": "redo",
    "Side view": "ui tool show 'Side View'"
}


def run_provider(session, name):
    what = _providers[name]
    if not isinstance(what, str):
        what(session)
    else:
        from chimerax.core.commands import run
        run(session, what)
