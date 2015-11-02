# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    # This function is simple because we "know" we only provide
    # a single tool in the entire package, so we do not need to
    # look at the name in 'tool_info.name'
    from . import cmd
    cmd.mousemodes_show(session)
    return cmd.get_singleton(session)


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name, tool_info):
    from . import cmd
    from chimera.core.commands import register
    register(command_name + " hide", cmd.hide_desc, cmd.mousemodes_hide)
    register(command_name + " show", cmd.show_desc, cmd.mousemodes_show)


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'MouseModePanel':
        from . import gui
        return gui.MouseModePanel
    return None
