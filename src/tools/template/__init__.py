# vi: set expandtab shiftwidth=4 softtabstop=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # If providing more than one tool in package,
    # look at the name in 'ti.name' to see which is being started.

    # Starting tools may only work in GUI mode, or in all modes.
    # If a tool instance is SESSION_ENDURING, then return the
    # singleton.
    from .gui import ToolUI
    return ToolUI(session, ti)     # UI should register itself with tool state manager


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    from . import cmd
    from chimera.core.commands import register
    register(command_name + " SUBCOMMAND_NAME",
             cmd.subcommand_desc, cmd.subcommand_function)
    # TODO: Register more subcommands here


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'ToolUI':
        from . import gui
        return gui.ToolUI
    return None
