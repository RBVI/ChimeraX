# vim: set expandtab ts=4 sw=4:

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

from .mousemodes import MouseMode, MouseModes, mod_key_info
from .std_modes import SelectMouseMode, select_pick, SelectContextMenuAction, \
                RotateMouseMode, TranslateMouseMode, RotateSelectedModelsMouseMode, \
                TranslateSelectedModelsMouseMode, ZoomMouseMode

from chimerax.core.toolshed import BundleAPI

class _MouseModesAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        from . import settings
        settings.settings = settings._MouseModesSettings(session, "mouse modes")
        settings.clip_settings = settings._MouseClipSettings(session, 'mouse clip')

        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready',
                lambda *args, ses=session: settings.register_settings_options(ses))

    @staticmethod
    def start_tool(session, tool_name):
        # 'start_tool' is called to start an instance of the tool
        from .tool import MouseModePanel
        return MouseModePanel.get_singleton(session)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command is lazily called when command is referenced
        from .cmd import register_mousemode_command
        register_mousemode_command(logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'MouseModePanel':
            from . import tool
            return tool.MouseModePanel
        return None

bundle_api = _MouseModesAPI()
