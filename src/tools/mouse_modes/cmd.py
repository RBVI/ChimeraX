# vi: set expandtab ts=4 sw=4:

def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    from .gui import MouseModePanel
    running = session.tools.find_by_class(MouseModePanel)
    if len(running) > 1:
        raise RuntimeError("Can only have one mouse mode panel")
    if not running:
        if create:
            tool_info = session.toolshed.find_tool('mouse_modes')
            return MouseModePanel(session, tool_info)
        else:
            return None
    else:
        return running[0]


def hide(session):
    mmpanel = get_singleton(session)
    if mmpanel is not None:
        mmpanel.display(False)
from chimera.core.commands import CmdDesc
hide_desc = CmdDesc()


def show(session):
    mmpanel = get_singleton(session, create=True)
    if mmpanel is not None:
        mmpanel.display(True)
show_desc = CmdDesc()
