# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
preferences: manage preferences
===============================

TODO
"""
from . import commands
from .colors import Color
from . import configfile
from .settings import Settings


class _CoreSettings(Settings):

    # if a new setting is to appear in the settings GUI, info must also be added to
    # chimerax.ui.core_settings_ui.py
    EXPLICIT_SAVE = {
        'background_color': configfile.Value(Color('#000'), commands.ColorArg, Color.hex_with_alpha),
        'http_proxy': ("", 80),
        'https_proxy': ("", 443),
        'resize_window_on_session_restore': False,
    }
    AUTO_SAVE = {
        'toolshed_update_interval': 'week',
        'toolshed_last_check': '',
        'newer_update_interval': 'week',
        'newer_last_check': '',
        'ignore_update': [],  # updates shown and then ignored
    }

def init(session):
    global settings
    settings = _CoreSettings(session, "chimerax.core")
    set_proxies(initializing=True)
    _init_background_color(session)

def set_proxies(*, initializing=False):
    import os
    for proxy_type in ("http", "https"):
        host, port = getattr(settings, proxy_type + "_proxy")
        environ_var = proxy_type + "_proxy"
        if host:
            os.environ[environ_var] = "%s://%s:%d" % (proxy_type, host, port)
        elif environ_var in os.environ and not initializing:
            del os.environ[environ_var]

def _init_background_color(session):
    if hasattr(session, 'main_view'):
        try:
            from .core_settings import settings
            session.main_view.background_color = settings.background_color.rgba
        except ImportError:
            # During ChimeraX build settings module may not yet be installed.
            pass
