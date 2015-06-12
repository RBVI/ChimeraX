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


class _Preferences(configfile.ConfigFile):

    PROPERTY_INFO = {
        'bg_color': configfile.Value(
            color.Color('#000'), color.ColorArg, color.Color.hex_with_alpha),
        'multisample_threshold': configfile.Value(
            0, cli.NonNegativeIntArg, str),
        'silhouette': False,
        # autostart map_series_gui until alternate means of installing
        # trigger is found
        'autostart': ['cmd_line', 'mouse_modes', 'log', 'sideview', 'map_series_gui'],
    }

    def __init__(self, session):
        configfile.ConfigFile.__init__(self, session, "chimera.core")


def get():
    assert(_prefs is not None)
    return _prefs


def init(session):
    global _prefs
    if _prefs is None:
        _prefs = _Preferences(session)
