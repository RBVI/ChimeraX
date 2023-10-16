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

from chimerax.core.colors import ColorValue, BuiltinColors
from chimerax.core.settings import Settings

class _DistanceSettings(Settings):
    EXPLICIT_SAVE = {
        'color': ColorValue(BuiltinColors['gold']),
        'dashes': 9,
        'decimal_places': 3,
        'radius': 0.1,
        'show_units': True,
    }

# 'settings' module attribute will be set by the initialization of the bundle API

def register_settings_options(session):
    from chimerax.core.colors import color_name
    from chimerax.ui.options import ColorOption, BooleanOption, IntOption, FloatOption
    settings_info = {
        'color': (
            "Color",
            ColorOption,
            "distance style color %s",
            color_name,
            "Color of atomic distance monitors"),
        'dashes': (
            "Number of dashes",
            (IntOption, {'min': 0 }),
            "distance style dashes %d",
            None,
            "How many dashes when drawing distance monitor.  Zero means solid line.  "
            "Currently, even numbers act the same as the next odd number."),
        'decimal_places': (
            "Decimal places",
            (IntOption, {'min': 0 }),
            "distance style decimalPlaces %d",
            None,
            "How many digits after the decimal point to show for distances"),
        'radius': (
            "Radius",
            (FloatOption, {'min': 'positive', 'decimal_places': 3 }),
            "distance style radius %g",
            None,
            "Radial line thickness of distance"),
        'show_units': (
            'Show angstrom symbol (\N{ANGSTROM SIGN})',
            BooleanOption,
            'distance style symbol %s',
            None,
            'Whether to show angstrom symbol after the distancee'),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, updater, converter, balloon = setting_info
        if isinstance(opt_class, tuple):
            opt_class, kw = opt_class
        else:
            kw = {}
        def _opt_cb(opt, updater=updater, converter=converter, ses=session):
            setting = opt.attr_name
            val = opt.value
            if converter:
                val = converter(val)
            from chimerax.core.commands import run
            run(ses, updater % val)
        opt = opt_class(opt_name, getattr(settings, setting), _opt_cb,
            attr_name=setting, settings=settings, balloon=balloon, auto_set_attr=False, **kw)
        session.ui.main_window.add_settings_option("Distances", opt)
