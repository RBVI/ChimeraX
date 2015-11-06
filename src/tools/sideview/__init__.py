# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    # If providing more than one tool in package,
    # look at the name in 'tool_info.name' to see which is being started.

    # Starting tools may only work in GUI mode, or in all modes.
    # Here, we check for GUI-only tool.
    if not session.ui.is_gui:
        return None
    from .gui import SideViewUI
    #return SideViewUI(session, tool_info)
    s = SideViewUI(session, tool_info)
    return s


# no commands


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'SideViewUI':
        from . import gui
        return gui.SideViewUI
    return None
