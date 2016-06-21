# vim: set expandtab ts=4 sw=4:
_initialized = False


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, bundle_info):
    if not session.ui.is_gui:
        return None
    from chimerax.core import window_sys
    if window_sys == "wx":
        return None	# Only Qt GUI supported
    from . import gui
    return gui.show_volume_dialog(session)

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'VolumeViewer':
        from . import gui
        return gui.VolumeViewer
    return None
