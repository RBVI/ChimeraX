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
    def start_tool(session, bundle_info):
        # 'start_tool' is called to start an instance of the tool
        # Starting tools may only work in GUI mode, or in all modes.
        from .tool import ToolshedUI
        return ToolshedUI.get_singleton(session)

    @staticmethod
    def register_command(command_name, bundle_info):
        # 'register_command' is lazily called when command is referenced
        from . import cmd
        from chimerax.core.commands import create_alias, register
        if command_name == "ts":
            create_alias("ts", "toolshed $*")
            return
        register(command_name + " list", cmd.ts_list_desc, cmd.ts_list)
        register(command_name + " refresh", cmd.ts_refresh_desc, cmd.ts_refresh)
        register(command_name + " install", cmd.ts_install_desc, cmd.ts_install)
        register(command_name + " remove", cmd.ts_remove_desc, cmd.ts_remove)
        # register(command_name + " update", cmd.ts_update_desc, cmd.ts_update)
        register(command_name + " start", cmd.ts_start_desc, cmd.ts_start)
        register(command_name + " show", cmd.ts_show_desc, cmd.ts_show)
        register(command_name + " hide", cmd.ts_hide_desc, cmd.ts_hide)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'ToolshedUI':
            from . import tool
            return tool.ToolshedUI
        return None

bundle_api = _MyAPI()
