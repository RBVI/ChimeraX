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

class _LabelSettings(Settings):
    EXPLICIT_SAVE = {
        'label_height': 0.7,
    }

# 'settings' module attribute will be set by the initialization of the bundle API

def register_settings_options(session):
    from chimerax.ui.options import BooleanOption, FloatOption
    settings_info = {
        'label_height': (
            '3D label height (\u212B)',
            FloatOption,
            'label defaultHeight %s',
            'Set default 3d label height'),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, updater, balloon = setting_info
        def _opt_cb(opt, updater=updater, ses=session):
            setting = opt.attr_name
            val = opt.value
            from chimerax.core.commands import run
            run(ses, updater % val)
        opt = opt_class(opt_name, getattr(settings, setting), _opt_cb,
            attr_name=setting, settings=settings, balloon=balloon, auto_set_attr=False)
        session.ui.main_window.add_settings_option("Labels", opt)
