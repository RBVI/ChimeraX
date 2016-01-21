# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, bundle_info):
    return get_singleton(session, create=True)

def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    from chimerax.core import tools
    from .gui import FilePanel
    return tools.get_singleton(session, FilePanel, 'file history', create=create)

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'FilePanel':
        from . import gui
        return gui.FilePanel
    return None
