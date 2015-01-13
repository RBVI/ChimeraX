# vim: set expandtab ts=4 sw=4:

#
# 'register_command' is called by the toolshed on start up
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # This function is simple because we "know" we only provide
    # a single tool in the entire package, so we do not need to
    # look at the name in 'ti.name'
    # For GUI, we create the graphical representation if it does
    # not already exist.
    # For all other types of UI, we do nothing.
    from chimera.core import gui
    if isinstance(session.ui, gui.UI):
        if not hasattr(session.ui, "cmd_line"):
            from .gui import CmdLine
            session.ui.cmd_line = CmdLine(session)

def register_command(command_name):
    from . import cmd
    from chimera.core import cli
    cli.register(command_name + " hide", cmd.hide_desc, cmd.hide)
    cli.register(command_name + " show", cmd.show_desc, cmd.show)
