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
        'autostart': [
            'molecule_display_shortcuts', 'log', 'model panel',
            'mouse_modes', 'graphics_shortcuts', 'cmd_line', 'cellpack'
        ],
        'atomspec_contents': 'simple', # choices: simple, command-line specifier, serial number
        'bg_color': configfile.Value(
            Color('#000'), commands.ColorArg, Color.hex_with_alpha),
        'multisample_threshold': configfile.Value(
            0, commands.NonNegativeIntArg, str),
        'silhouette': False,
    }

def init(session):
    global settings
    settings = _CoreSettings(session, "chimerax.core")
