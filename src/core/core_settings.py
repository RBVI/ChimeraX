# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
preferences: manage preferences
===============================

TODO
"""
from . import commands
from .colors import Color
from . import configfile
from .settings import Settings
_prefs = None


class _CoreSettings(Settings):

    EXPLICIT_SAVE = {
        'bg_color': configfile.Value(
            Color('#000'), commands.ColorArg, Color.hex_with_alpha),
        'multisample_threshold': configfile.Value(
            0, commands.NonNegativeIntArg, str),
        'silhouette': False,
        # autostart map_series_gui until alternate means of installing
        # trigger is found
        'autostart': ['molecule_display_shortcuts', 'log', 'model panel',
                'mouse_modes', 'graphics_shortcuts', 'cmd_line', 'map_series_gui'],
    }

def init(session):
    global settings
    settings = _CoreSettings(session, "chimera.core")
