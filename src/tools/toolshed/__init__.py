# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    # Starting tools may only work in GUI mode, or in all modes.
    from .gui import ToolshedUI
    return ToolshedUI.get_singleton(session)


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, tool_info):
    from . import cmd
    from chimera.core.commands import create_alias, register
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


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'ToolshedUI':
        from . import gui
        return gui.ToolshedUI
    return None
