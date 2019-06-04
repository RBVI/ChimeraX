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


from  chimerax.core.settings import Settings

class _CmdLineSettings(Settings):
    EXPLICIT_SAVE = {
        'startup_commands': [],
    }
    AUTO_SAVE = {
        "num_remembered": 500,
        "typed_only": True,
    }

# 'settings' module attribute will be set by the initialization of the bundle API

def register_settings_options(session):
    from chimerax.ui.options import StringsOption
    settings_info = {
        'startup_commands': (
            "Execute these commands at startup",
            StringsOption,
            "List of commands to execute when ChimeraX command-line tool starts"),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        opt = opt_class(opt_name, getattr(settings, setting), None,
            attr_name=setting, settings=settings, balloon=balloon)
        session.ui.main_window.add_settings_option("Startup", opt)
