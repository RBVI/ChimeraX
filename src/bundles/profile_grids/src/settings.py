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

from chimerax.ui.options import Option, BooleanOption, IntOption, \
    OptionalRGBAOption, OptionalRGBAPairOption

APPEARANCE = "Appearance"

defaults = {
    "percent_decimal_places": (APPEARANCE,
        "Decimal places for percentages", 1, IntOption, {'min': 0, 'max': 3}, 0),
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _PGSettings(Settings):
    EXPLICIT_SAVE = { k: v[-1] for k, v in defaults.items() }

def init(session):
    # each Profile Grids instance has its own settings instance
    return _PGSettings(session, "Profile Grids")
