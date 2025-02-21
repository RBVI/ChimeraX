# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
ui: ChimeraX user interface
===========================
"""

from .htmltool import HtmlToolInstance
from .font import shrink_font, set_line_edit_width
from .gui import MainToolWindow, initialize_qt, menu_capitalize, tool_user_error

from chimerax.core.toolshed import BundleAPI

class _UIBundleAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command is lazily called when command is referenced
        from .cmd import register_ui_command, register_tool_command
        if command_name.startswith('tool'):
            register_tool_command(logger)
        else:
            register_ui_command(logger)

bundle_api = _UIBundleAPI()
