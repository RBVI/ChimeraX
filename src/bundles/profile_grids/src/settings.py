# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.ui.options import Option, BooleanOption, IntOption, EnumOption

APPEARANCE = "Appearance"

defaults = {
    "cell_text": (APPEARANCE,
        "Cell text", 1, EnumOption, { 'values': ["none", "count", "percentage"] }, "percentage"),
    "percent_decimal_places": (APPEARANCE,
        "Decimal places for percentages", 2, IntOption, {'min': 0, 'max': 3}, 0),
}

prevalence_defaults = {
    "prevalence_main_color_info": (
        # RdBu3 color palette
        True, [(0.0, (103, 169, 207, 255)), (1.0, (247, 247, 247, 255)), (2.0, (239, 138, 98, 255))],
        True, 0.5, "dark gray",
        True
    ),
    "prevalence_chosen_color_info": (True, "white"),
    "prevalence_unchosen_color_info": (True, "dark gray"),
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _PGSettings(Settings):
    EXPLICIT_SAVE = { k: v[-1] for k, v in defaults.items() }
    AUTO_SAVE = { k: v for k, v in prevalence_defaults.items() }

def init(session):
    # each Profile Grids instance has its own settings instance
    return _PGSettings(session, "Profile Grids")
