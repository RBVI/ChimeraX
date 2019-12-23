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

from chimerax.core.colors import Color, BuiltinColors
from chimerax.core import configfile, commands
from chimerax.core.settings import Settings

class _PresetsSettings(Settings):
    EXPLICIT_SAVE = {
        'folder': None,
    }

# 'settings' module attribute will be set by manager initialization

def register_settings_options(session):
    from chimerax.ui.options import InputFolderOption
    settings_info = {
        'folder': (
            "Custom presets folder",
            InputFolderOption,
            "Folder containing definitions for personal presets, which will appear in the Presets menu"),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        opt = opt_class(opt_name, getattr(settings, setting), None,
            attr_name=setting, settings=settings, balloon=balloon)
        session.ui.main_window.add_settings_option("Startup", opt)
