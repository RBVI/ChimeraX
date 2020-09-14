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

from chimerax.core.settings import Settings

class _LogSettings(Settings):
    EXPLICIT_SAVE = {
        'errors_raise_dialog': True,
        'warnings_raise_dialog': False,
        'session_restore_clears': True,
    }

    AUTO_SAVE = {
        "exec_cmd_links": False,
    }

# 'settings' module attribute will be set by the initialization of the bundle API

def register_settings_options(session):
    from chimerax.ui.options import BooleanOption
    from chimerax.core.utils import CustomSortString
    settings_info = {
        'errors_raise_dialog': (
            CustomSortString('Errors shown in dialog', 1),
            BooleanOption,
            'Should error messages be shown in a separate dialog as well as being logged'),
        'warnings_raise_dialog': (
            CustomSortString('Warnings shown in dialog', 1),
            BooleanOption,
            'Should warning messages be shown in a separate dialog as well as being logged'),
        'session_restore_clears': (
            CustomSortString('Restoring session clears log', 2),
            BooleanOption,
            'Restoring session clears log'),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        opt = opt_class(opt_name, getattr(settings, setting), None,
            attr_name=setting, settings=settings, balloon=balloon)
        session.ui.main_window.add_settings_option("Log", opt)
