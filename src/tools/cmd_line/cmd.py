# vim: set expandtab ts=4 sw=4:

from chimera.core import cli

def hide(session):
    session.ui.cmd_line.tool_window.shown = False
hide_desc = cli.CmdDesc()
