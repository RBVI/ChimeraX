# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # Starting tools may only work in GUI mode, or in all modes.
    # Here, we check for GUI-only tool.
    from chimera.core import gui
    if isinstance(session.ui, gui.UI):
        if not hasattr(session.ui, ti.name):
            from .gui import ToolshedUI
            setattr(session.ui, ti.name, ToolshedUI(session))


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    from . import cmd
    from chimera.core import cli
    if command_name == "ts":
        cli.alias(None, "ts", "toolshed $*")
        return
    cli.register(command_name + " list", cmd.ts_list_desc, cmd.ts_list)
    cli.register(command_name + " refresh", cmd.ts_refresh_desc, cmd.ts_refresh)
    cli.register(command_name + " install", cmd.ts_install_desc, cmd.ts_install)
    cli.register(command_name + " remove", cmd.ts_remove_desc, cmd.ts_remove)
    # cli.register(command_name + " update", cmd.ts_update_desc, cmd.ts_update)
