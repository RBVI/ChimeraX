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

class _MouseModesSettings(Settings):
    EXPLICIT_SAVE = {
        'trackpad_multitouch': True,
        'trackpad_sensitivity': 1.0,
    }

# 'settings' module attribute will be set by the initialization of the bundle API

def register_settings_options(session):
    from chimerax.ui.options import BooleanOption, FloatOption
    settings_info = {
        'trackpad_multitouch': (
            'Trackpad gestures to rotate and move',
            BooleanOption,
            _enable_trackpad_multitouch,
            'Whether to enable 2 and 3 finger trackpad drags to rotate and move.'),
        'trackpad_sensitivity': (
            'Trackpad sensitivity',
            (FloatOption, {'decimal_places': 2 }),
            _set_trackpad_sensitivity,
            'How fast models move in response to multitouch trackpad gestures'),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, updater, balloon = setting_info
        if isinstance(opt_class, tuple):
            opt_class, kw = opt_class
        else:
            kw = {}
        def _opt_cb(opt, updater=updater, ses=session):
            updater(ses, opt.value)
        opt = opt_class(opt_name, getattr(settings, setting), _opt_cb,
            attr_name=setting, settings=settings, balloon=balloon, **kw)
        session.ui.main_window.add_settings_option("Trackpad", opt)

def _enable_trackpad_multitouch(session, enable):
    session.ui.mouse_modes.trackpad.enable_multitouch(enable)

def _set_trackpad_sensitivity(session, value):
    session.ui.mouse_modes.trackpad.trackpad_speed = value

