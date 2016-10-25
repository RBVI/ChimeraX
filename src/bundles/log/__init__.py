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

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'Log':
            from . import tool
            return tool.Log
        return None

    @staticmethod
    def start_tool(session, tool_name):
        # 'start_tool' is called to start an instance of the tool
        from . import cmd
        return cmd.get_singleton(session, create=True)

    @staticmethod
    def register_command(command_name):
        # 'register_command' is lazily called when command is referenced
        from . import cmd
        from chimerax.core.commands import register, create_alias
        if command_name == "echo":
            create_alias("echo", "log text $*")
            return
        register(command_name, cmd.log_desc, cmd.log)

bundle_api = _MyAPI()
