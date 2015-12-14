# vim: set expandtab shiftwidth=4 softtabstop=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, bundle_info):
    # If providing more than one tool in package,
    # look at the name in 'bundle_info.name' to see which is being started.

    # Starting tools may only work in GUI mode, or in all modes.
    # If a tool instance is SESSION_ENDURING, then return the
    # singleton.
    if not session.ui.is_gui:
        return
    from .gui import ShellUI
    return ShellUI(session, bundle_info)     # UI should register itself with tool state manager


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, bundle_info):
    pass


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'ShellUI':
        from . import gui
        return gui.ShellUI
    return None
