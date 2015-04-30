# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
preferences: manage preferences
===============================

TODO
"""
from . import cli
from . import color
from . import configfile
_prefs = None


class _Graphics(configfile.Section):

    PROPERTY_INFO = {
        'bg_color': configfile.Value(
            color.Color('#000'), color.ColorArg, color.Color.hex_with_alpha),
        'multisample_threshold': configfile.Value(
            120, cli.NonNegativeIntArg, str),
        'silhouette': False,
    }


class _Preferences(configfile.ConfigFile):

    def __init__(self, session):
        configfile.ConfigFile.__init__(self, session, "chimera.core")
        self.graphics = _Graphics(self, 'graphics')


def get():
    return _prefs


def init(session):
    global _prefs
    if _prefs is None:
        _prefs = _Preferences(session)
        session.triggers.add_handler(_prefs.trigger_name(), _update_prefs)
    return _prefs


def _update_prefs(trigger, data):
    # monitor preference changes, and update the right places
    session, section, name, value = data
    if section == 'graphics':
        view = session.main_view
        if name == 'bg_color':
            view.background_color = value.rgba
            view.redraw_needed = True
        elif name == 'shilhouette':
            view.silhouettes = value
            view.redraw_needed = True


def set_cmd(session, bg_color=None, silhouette=None, multisample_threshold=None):
    prefs = get()
    if bg_color is not None:
        prefs.graphics.bg_color = bg_color
    if silhouette is not None:
        prefs.graphics.silhouettes = silhouette
    if multisample_threshold is not None:
        prefs.graphics.multisample_threshold = multisample_threshold

_set_desc = cli.CmdDesc(
    keyword=[('bg_color', color.ColorArg),
             ('silhouette', cli.BoolArg),
             ('multisample_threshold', cli.NonNegativeIntArg), ]
)


def register_set_command():
    cli.register('set', _set_desc, set_cmd)
