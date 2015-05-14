# vi: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # If providing more than one tool in package,
    # look at the name in 'ti.name' to see which is being started.

    # Starting tools may only work in GUI mode, or in all modes.
    # Here, we check for GUI-only tool.
    if not session.ui.is_gui:
        return None
    from .gui import SideViewUI
    return SideViewUI(session, ti)


# no commands
