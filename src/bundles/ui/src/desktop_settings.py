# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
GNOME appearance settings without Qt's GTK3 platform theme
===========================================================

On GNOME, Qt loads its GTK3 platform-theme plugin, which loads GTK and runs
gtk_init(): about 30 ms of every ChimeraX start.  Qt's built-in "gnome"
theme gives the same dialog button order without GTK, but not the desktop's
icon theme (toolbar icons come from QIcon.fromTheme), font, or dark mode
preference.  This module reads those from GSettings, which reads the dconf
database directly with no D-Bus round trip (about 1 ms), and applies them.

Native GTK file dialogs are not available without the GTK3 theme; Qt's own
dialogs are used instead, as ChimeraX's Open dialog already does.

Set QT_QPA_PLATFORMTHEME (for example to gtk3) to use a platform theme
instead.
"""

import os
import sys

# Desktops whose appearance is in the org.gnome.desktop.interface schema.
_GNOME_DESKTOPS = {"gnome", "ubuntu", "unity"}
_SCHEMA = b"org.gnome.desktop.interface"
_KEYS = ("color-scheme", "gtk-theme", "icon-theme", "font-name")

_settings = None		# Values read from GSettings, if the gnome theme is used
_gio = None
_gsettings = None
_changed_callback = None	# Keep the ctypes callback alive


def select_platform_theme():
    """Before the QApplication is created, use Qt's built-in gnome platform
    theme on a GNOME desktop if its settings can be read.  Returns whether it did.
    """
    global _settings
    if not sys.platform.startswith("linux") or "QT_QPA_PLATFORMTHEME" in os.environ:
        return False
    desktops = {d.strip().lower() for d in os.environ.get("XDG_CURRENT_DESKTOP", "").split(":")}
    if not desktops & _GNOME_DESKTOPS:
        return False
    try:
        settings = _read_settings()
    except Exception:
        return False
    if settings is None:
        return False
    _settings = settings
    os.environ["QT_QPA_PLATFORMTHEME"] = "gnome"
    return True


def active():
    """Whether the gnome platform theme is used with settings from GSettings."""
    return _settings is not None


def _read_settings():
    import ctypes
    global _gio, _gsettings
    gio = ctypes.CDLL("libgio-2.0.so.0")
    gio.g_settings_schema_source_get_default.restype = ctypes.c_void_p
    gio.g_settings_schema_source_lookup.restype = ctypes.c_void_p
    gio.g_settings_schema_source_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    gio.g_settings_schema_has_key.restype = ctypes.c_int
    gio.g_settings_schema_has_key.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    gio.g_settings_schema_unref.argtypes = [ctypes.c_void_p]
    gio.g_settings_new.restype = ctypes.c_void_p
    gio.g_settings_new.argtypes = [ctypes.c_char_p]
    gio.g_settings_get_string.restype = ctypes.c_void_p
    gio.g_settings_get_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    gio.g_free.argtypes = [ctypes.c_void_p]
    source = gio.g_settings_schema_source_get_default()
    if not source:
        return None
    # g_settings_new() aborts the process for an unknown schema, so check first.
    schema = gio.g_settings_schema_source_lookup(source, _SCHEMA, 1)
    if not schema:
        return None
    keys = [k for k in _KEYS if gio.g_settings_schema_has_key(schema, k.encode())]
    gio.g_settings_schema_unref(schema)
    if "icon-theme" not in keys or "font-name" not in keys:
        return None
    gsettings = gio.g_settings_new(_SCHEMA)
    if not gsettings:
        return None
    _gio, _gsettings = gio, gsettings
    return {key: _get_string(key) for key in keys}


def _get_string(key):
    import ctypes
    p = _gio.g_settings_get_string(_gsettings, key.encode())
    value = ctypes.string_at(p).decode("utf-8", "replace")
    _gio.g_free(p)
    return value


def prefers_dark():
    """Whether the desktop prefers dark windows, as the GTK3 theme decides."""
    if _settings is None:
        return False
    scheme = _settings.get("color-scheme", "default")
    if scheme == "prefer-dark":
        return True
    if scheme == "prefer-light":
        return False
    return _settings.get("gtk-theme", "").lower().endswith("-dark")


def parse_font(description):
    """Family and point size from a Pango font description like "Ubuntu Sans 11"."""
    parts = description.rsplit(None, 1)
    if len(parts) == 2:
        try:
            return parts[0].rstrip(","), float(parts[1])
        except ValueError:
            pass
    return description, None


def apply_settings(app):
    """After the QApplication is created, use the desktop's icon theme and
    font, and follow its dark mode preference while ChimeraX runs."""
    if _settings is None:
        return
    from Qt.QtGui import QIcon, QFont
    QIcon.setThemeName(_settings["icon-theme"])
    QIcon.setFallbackThemeName("hicolor")
    family, size = parse_font(_settings["font-name"])
    font = QFont(family)
    if size:
        font.setPointSizeF(size)
    app.setFont(font)
    _watch_color_scheme(app)


def _watch_color_scheme(app):
    # GSettings reports changes through the GLib main loop, which Qt's event
    # loop runs on Linux.
    import ctypes
    global _changed_callback
    try:
        gobject = ctypes.CDLL("libgobject-2.0.so.0")
    except OSError:
        return
    proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)

    def changed(gsettings, key, data):
        try:
            key = key.decode()
            _settings[key] = _get_string(key)
            if getattr(app, "following_desktop_color_scheme", False):
                app._change_color_scheme('dark' if prefers_dark() else 'light')
        except Exception:
            pass

    _changed_callback = proto(changed)
    gobject.g_signal_connect_data.restype = ctypes.c_ulong
    gobject.g_signal_connect_data.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    for key in ("color-scheme", "gtk-theme"):
        if key in _settings:
            gobject.g_signal_connect_data(_gsettings, b"changed::" + key.encode(),
                                          ctypes.cast(_changed_callback, ctypes.c_void_p),
                                          None, None, 0)
