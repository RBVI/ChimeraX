# vi: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # This function is simple because we "know" we only provide
    # a single tool in the entire package, so we do not need to
    # look at the name in 'ti.name'
    from . import gui
    spanel = gui.get_singleton(session, create = True)
    if spanel is not None:
        spanel.display(True)
    return spanel


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    from . import shortcuts
    shortcuts.register_shortcut_command()
