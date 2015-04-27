# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
preferences: manage preferences
===============================

TODO
"""
from . import color
from . import configinfo
_prefs = None


class _Graphics:

    PROPERTY_INFO = {
        'bg_color': configinfo.Value(color.Color((0, 0, 0, 0)),
                                     color.ColorArg, color.Color.hex)
    }


class _Preferences(configinfo.ConfigInfo):

    def __init__(self, session):
        configinfo.ConfigInfo.__init__(self, "chimera.core")
        self.graphics = _Graphics(self, 'graphics')


def get():
    global _prefs
    if _prefs is None:
        _prefs = _Preferences()
    return _prefs


def init(app_dirs):
    pass
