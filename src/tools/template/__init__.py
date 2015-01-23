# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # If providing more than one tool in package,
    # look at the name in 'ti.name' to see which is being started.

    # Starting tools may only work in GUI mode, or in all modes.
    # Here, we check for GUI-only tool.
    from chimera.core import gui
    if isinstance(session.ui, gui.UI):
        if not hasattr(session.ui, ti.name):
            from .gui import ToolUI
            setattr(session.ui, ti.name, ToolUI(session))


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    from . import cmd
    from chimera.core import cli
    cli.register(command_name + " SUBCOMMAND_NAME",
                 cmd.subcommand_desc, cmd.subcommand_function)
    # TODO: Register more subcommands here
