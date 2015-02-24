# vi: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    # GUI actually starts when data opened.
    # Just register callback to detect map series open here.
    from . import gui
    gui.show_slider_on_open(session)

#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    pass        # No command
