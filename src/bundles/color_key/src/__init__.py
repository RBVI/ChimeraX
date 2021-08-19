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

from chimerax.core.toolshed import BundleAPI
from .cmd import show_key

class _ColorKeyBundle(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called (when the command is run)
        from .cmd import register_command
        register_command(logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'ColorKeyModel':
            from .model import ColorKeyModel
            return ColorKeyModel
        return None

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import ColorKeyTool
        return ColorKeyTool(session, tool_name)

bundle_api = _ColorKeyBundle()
