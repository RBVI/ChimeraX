# vim: set expandtab ts=4 sw=4:

#
# 'register_command' is called by the toolshed on start up
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # If providing more than one tool in package,
    # look at the name in 'ti.name' to see which is being started.
    session.logger.error("Toolshed UI not implemented yet")
    return

    # Starting tools may only work in GUI mode, or in all modes.
    # Here, we check for GUI-only tool.
    from chimera.core import gui
    if isinstance(session.ui, gui.UI):
        if not hasattr(session.ui, ti.name):
            from .gui import ToolUI
            setattr(session.ui, ti.name, ToolUI(session))

def register_command(command_name):
    import sys
    print("toolshed.register_command:", command_name, file=sys.stderr)
    from . import cmd
    from chimera.core import cli
    cli.register(command_name + " list", cmd.ts_list_desc, cmd.ts_list)
    cli.register(command_name + " refresh", cmd.ts_refresh_desc, cmd.ts_refresh)
    #cli.register(command_name + " update", cmd.ts_update_desc, cmd.ts_update)
    #cli.register(command_name + " install", cmd.ts_install_desc, cmd.ts_install)
