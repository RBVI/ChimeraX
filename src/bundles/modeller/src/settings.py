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

class _ModellerSettings(Settings):

    AUTO_SAVE = {
        'combine_templates': False,
        'fast': False,
        'het_preserve': False,
        'hydrogens': False,
        'license_key': None,
        'num_models': 5,
        'temp_path': "",
        'water_preserve': False
    }

settings = None
def get_settings(session):
    global settings
    if settings is None:
        settings = _ModellerSettings(session, "modeller")
    return settings
