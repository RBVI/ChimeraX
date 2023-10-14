# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .swap_res import swap_aa, get_rotamers

from chimerax.core.toolshed import BundleAPI

class SwapResAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "_RotamerStateManager":
            from .cmd import _RotamerStateManager
            return _RotamerStateManager
        elif class_name == "RotamerDialog":
            from .tool import RotamerDialog
            return RotamerDialog

    @staticmethod
    def initialize(session, bundle_info):
        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready', lambda *args, ses=session:
                SwapResAPI._add_gui_items(ses))

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(command_name, logger)

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import prep_rotamers_dialog
        return prep_rotamers_dialog(session, tool_name)

    @staticmethod
    def _add_gui_items(session):
        from .contextmenu import add_selection_context_menu_items
        add_selection_context_menu_items(session)

bundle_api = SwapResAPI()
