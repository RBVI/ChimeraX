# vi: set expandtab ts=4 sw=4:
_initialized = False


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, ti):
    global _initialized
    if not session.ui.is_gui:
        return None
    # GUI actually starts when data is opened, so this is for
    # restoring sessions
    from . import gui
    if not _initialized:
        # Called first time during autostart.
        # Just register callback to detect map series open here.
        gui.show_slider_on_open(session)
        _initialized = True
        return None
    else:
        return gui.MapSeries(session, ti)


#
# 'register_command' is called by the toolshed on start up
#
def register_command(command_name):
    pass        # No command


#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'MapSeries':
        from . import gui
        return gui.MapSeries
    return None
