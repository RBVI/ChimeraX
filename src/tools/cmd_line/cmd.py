# vi: set expandtab ts=4 sw=4:

from chimera.core.commands import CmdDesc


def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    from .gui import CommandLine
    running = session.tools.find_by_class(CommandLine)
    if len(running) > 1:
        raise RuntimeError("too many command line instances running")
    if not running:
        if create:
            tool_info = session.toolshed.find_tool('cmd_line')
            return CommandLine(session, tool_info)
        else:
            return None
    else:
        return running[0]


def hide(session):
    cmdline = get_singleton(session)
    if cmdline is not None:
        cmdline.display(False)
hide_desc = CmdDesc()


def show(session):
    cmdline = get_singleton(session, create=True)
    if cmdline is not None:
        cmdline.display(True)
show_desc = CmdDesc()
