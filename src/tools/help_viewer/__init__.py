# vi: set expandtab shiftwidth=4 softtabstop=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    # Starting tools may only work in GUI mode, or in all modes.
    # Here, we check for GUI-only tool.
    if not session.ui.is_gui:
        return None
    from .gui import get_singleton
    return get_singleton(session)


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, tool_info):
    from . import cmd
    from chimera.core.commands import register
    register(command_name, cmd.help_desc, cmd.help)


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'HelpUI':
        from . import gui
        return gui.HelpUI
    return None
