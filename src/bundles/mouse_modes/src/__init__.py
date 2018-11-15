# vim: set expandtab ts=4 sw=4:

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

from .mousemodes import MouseMode, MouseModes, mod_key_info
from .mousemodes import picked_object, picked_object_on_segment
from .std_modes import SelectMouseMode, select_pick, SelectContextMenuAction, \
                RotateMouseMode, TranslateMouseMode, RotateSelectedMouseMode, \
                TranslateSelectedMouseMode, ZoomMouseMode

from chimerax.core.toolshed import BundleAPI

class _MouseModesAPI(BundleAPI):

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
