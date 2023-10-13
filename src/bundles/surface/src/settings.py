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

class _SurfaceSettings(Settings):
    EXPLICIT_SAVE = {
        'clipping_surface_caps': True,
        'clipping_cap_offset': 0.01,
        'clipping_cap_subdivision': 1.0,
        'clipping_cap_on_mesh': False,
    }

# 'settings' module attribute will be set by the initialization of the bundle API

def register_settings_options(session):
    from chimerax.ui.options import BooleanOption, FloatOption
    settings_info = {
        'clipping_surface_caps': (
            'Cap clipped surfaces',
            BooleanOption,
            'surface cap %s',
            'Whether to cap surface holes created by clipping'),
        'clipping_cap_subdivision': (
            'Cap subdivision factor',
            FloatOption,
            'surface cap subdivision %s',
            'How fine to make the triangles on clip caps, for multi-coloring'),
        'clipping_cap_on_mesh': (
            'Cap mesh surfaces',
            BooleanOption,
            'surface cap mesh %s',
            'Whether to cap mesh style surfaces'),
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
        session.ui.main_window.add_settings_option("Clipping", opt)
