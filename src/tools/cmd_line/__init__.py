# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    from .gui import CommandLine
    return CommandLine.get_singleton(session)

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'CommandLine':
        from . import gui
        return gui.CommandLine
    return None
