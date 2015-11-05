# vim: set expandtab ts=4 sw=4:


#
# 'start_tool' is called to start an instance of the tool
#
def start_tool(session, tool_info):
    return get_singleton(session, create=True, display=True)

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'MouseModePanel':
        from . import gui
        return gui.MouseModePanel
    return None

def get_singleton(session, create=False, display=False):
    if not session.ui.is_gui:
        return None
    from .gui import MouseModePanel
    running = session.tools.find_by_class(MouseModePanel)
    if len(running) > 1:
        raise RuntimeError("Can only have one mouse mode panel")
    if not running:
        if create:
            tool_info = session.toolshed.find_tool('mouse_modes')
            tinst = MouseModePanel(session, tool_info)
        else:
            tinst = None
    else:
        tinst = running[0]
    if display and tinst:
        tinst.display(True)
    return tinst
