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
    return _prefs
