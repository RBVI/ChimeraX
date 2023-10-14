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

from chimerax.core.settings import Settings

class _MouseModesSettings(Settings):
    EXPLICIT_SAVE = {
        'trackpad_multitouch': True,
        'trackpad_sensitivity': 1.0,
        'trackpad_twist_speed': 2.0,
    }

# 'settings' and 'clip_settings' module attributes will be set by the initialization of the bundle API

def register_settings_options(session):
    register_trackpad_settings(session)
    register_clip_settings(session)

def register_trackpad_settings(session):    
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
        'trackpad_twist_speed': (
            'Trackpad twist speed',
            (FloatOption, {'decimal_places': 2 }),
            _set_trackpad_twist_speed,
            'How fast models rotate in response to multitouch 2-finger twist'),
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

def _set_trackpad_twist_speed(session, value):
    session.ui.mouse_modes.trackpad._twist_scaling = value

class _MouseClipSettings(Settings):
    EXPLICIT_SAVE = {
        'mouse_clip_plane_type': 'scene planes',
    }

def register_clip_settings(session):
    from chimerax.ui.options import SymbolicEnumOption
    class ClipPlaneTypeOption(SymbolicEnumOption):
        values = ('scene planes', 'screen planes')
        labels = ('scene planes', 'screen planes')

    settings_info = {
        'mouse_clip_plane_type': (
            'Mouse clipping enables',
            ClipPlaneTypeOption,
            'Does clipping mouse mode enable planes with fixed scene position or always perpendicular to screen.?'
            ),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        opt = opt_class(opt_name, getattr(clip_settings, setting), None,
            attr_name=setting, settings=clip_settings, balloon=balloon)
        session.ui.main_window.add_settings_option("Clipping", opt)
