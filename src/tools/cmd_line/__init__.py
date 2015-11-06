# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    # This function is simple because we "know" we only provide
    # a single tool in the entire package, so we do not need to
    # look at the name in 'tool_info.name'
    return get_singleton(session, create=True, display=True)

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'CommandLine':
        from . import gui
        return gui.CommandLine
    return None

def get_singleton(session, create=False, display=False):
    if not session.ui.is_gui:
        return None
    from .gui import CommandLine
    running = session.tools.find_by_class(CommandLine)
    if len(running) > 1:
        raise RuntimeError("too many command line instances running")
    if not running:
        if create:
            tool_info = session.toolshed.find_tool('cmd_line')
            tinst = CommandLine(session, tool_info)
        else:
            tinst = None
    else:
        tinst = running[0]
    if display and tinst:
        tinst.display(True)
    return tinst
