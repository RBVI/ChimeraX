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

class _ModellerComparativeSettings(Settings):

    AUTO_SAVE = {
        'fast': False,
        'het_preserve': False,
        'hydrogens': False,
        'license_key': None,
        'multichain': True,
        'num_models': 5,
        'temp_path': "",
        'water_preserve': False
    }

from .loops import ALL_MISSING
class _ModellerLoopsSettings(Settings):

    AUTO_SAVE = {
        'adjacent_flexible': 1,
        'num_models': 5,
        'protocol': "standard",
        'region': ALL_MISSING,
        'temp_path': "",
    }

_comparative_settings = _loops_settings = None
def get_settings(session, settings_type):
    global _loops_settings, _comparative_settings
    if settings_type == "Modeller Comparative" or settings_type == "license":
        if _comparative_settings is None:
            _comparative_settings = _ModellerComparativeSettings(session, "modeller")
        settings = _comparative_settings
    else:
        if _loops_settings is None:
            _loops_settings = _ModellerLoopsSettings(session, "modeller loops")
        settings = _loops_settings
    return settings
