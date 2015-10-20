# vi: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    # GUI started by command, so this is for restoring sessions
    if not session.ui.is_gui:
        return None
    from .gui import Plot
    return Plot(session, tool_info)


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, tool_info):
    from chimera.core.commands import register
    from . import cmd
    register('contacts', cmd.contacts_desc, cmd.contacts)


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'Plot':
        from . import gui
        return gui.Plot
    return None
