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

class _DistanceSettings(Settings):
    EXPLICIT_SAVE = {
        'color': configfile.Value(BuiltinColors['gold'], commands.ColorArg, Color.hex_with_alpha),
        'dashes': 9,
        'decimal_places': 3,
        'radius': 0.1,
        'show_units': True,
    }

# 'settings' module attribute will be set by the initialization of the bundle API
