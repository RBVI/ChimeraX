# vim: set expandtab ts=4 sw=4:

from chimera.core import cli

def hide(session):
    from . import _instances
    if session in _instances:
        _instances[session].tool_window.shown = False
hide_desc = cli.CmdDesc()

def show(session):
    from . import _instances
    if session in _instances:
        _instances[session].tool_window.shown = True
show_desc = cli.CmdDesc()
