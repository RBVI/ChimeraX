# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
preferences: manage preferences
===============================

TODO
"""
from . import commands
from .colors import Color, BuiltinColors
from . import configfile
from .settings import Settings


class _CoreSettings(Settings):

    # if a new setting is to appear in the settings GUI, info must also be added to
    # chimerax.ui.core_settings_ui.py
    EXPLICIT_SAVE = {
        'atomspec_contents': 'simple', # choices: simple, command (-line specifier), serial (number)
        'bg_color': configfile.Value(Color('#000'), commands.ColorArg, Color.hex_with_alpha),
        'clipping_surface_caps': True,
        'clipping_cap_offset': 0.01,
        'default_tool_window_side': "right",
        'http_proxy': ("", 80),
        'https_proxy': ("", 443),
        'initial_window_size': ("last used", None),
        'resize_window_on_session_restore': False,
    }
    AUTO_SAVE = {
        'distance_color': configfile.Value(BuiltinColors['gold'], commands.ColorArg, Color.hex_with_alpha),
        'distance_dashes': 9,
        'distance_decimal_places': 3,
        'distance_radius': 0.1,
        'distance_show_units': True,
        'last_window_size': None,
        'toolshed_update_interval': 'week',
        'toolshed_last_check': '',
    }

def init(session):
    global settings
    settings = _CoreSettings(session, "chimerax.core")
    set_proxies(initializing=True)

def set_proxies(*, initializing=False):
    import os
    for proxy_type in ("http", "https"):
        host, port = getattr(settings, proxy_type + "_proxy")
        environ_var = proxy_type + "_proxy"
        if host:
            os.environ[environ_var] = "%s://%s:%d" % (proxy_type, host, port)
        elif environ_var in os.environ and not initializing:
            del os.environ[environ_var]

