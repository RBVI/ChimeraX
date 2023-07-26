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

chain_area_default = 300
residue_area_default = 15

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        # 'start_tool' is called to start an instance of the tool
        # GUI started by command, so this is for restoring sessions
        from .tool import Plot
        return Plot(session, tool_name)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is called by the toolshed on start up
        from . import cmd
        cmd.register_interfaces(logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'Plot':
            from . import tool
            return tool.Plot
        return None

bundle_api = _MyAPI()
