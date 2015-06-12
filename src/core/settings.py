# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
settings: manage settings
=========================

TODO
"""
from . import cli
from . import color


def _set_cmd(session, bg_color=None, silhouettes=None):
    had_arg = False
    view = session.main_view
    if bg_color is not None:
        had_arg = True
        view.background_color = bg_color.rgba
        view.redraw_needed = True
    if silhouettes is not None:
        had_arg = True
        view.silhouettes = silhouettes
        view.redraw_needed = True
    if had_arg:
        return
    print('Current settings:\n'
          '  bg_color:', view.background_color, '\n'
          '  silhouettes:', view.silhouettes, '\n')

_set_desc = cli.CmdDesc(
    keyword=[('bg_color', color.ColorArg),
             ('silhouettes', cli.BoolArg)],
    synopsis="set preferences"
)


def register_set_command():
    cli.register('set', _set_desc, _set_cmd)
