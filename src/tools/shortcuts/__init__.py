# vim: set expandtab ts=4 sw=4:

#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    # This function is simple because we "know" we only provide
    # a single tool in the entire package, so we do not need to
    # look at the name in 'tool_info.name'
    from . import gui
    spanel = gui.get_singleton(tool_info.name, session, create = True)
    if spanel is not None:
        spanel.display(True)

    # TODO: Is there a better place to register selectors?
    from . import shortcuts
    shortcuts.register_selectors(session)

    return spanel


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, tool_info):
    from . import shortcuts
    shortcuts.register_shortcut_command()


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'ShortcutPanel':
        from . import gui
        return gui.ShortcutPanel
    return None
