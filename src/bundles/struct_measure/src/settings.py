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

class _AngleSettings(Settings):

    EXPLICIT_SAVE = {
        'decimal_places': 1,
    }

_angle_settings = None
def get_settings(session, settings_type):
    global _angle_settings
    if settings_type == "angles":
        if _angle_settings is None:
            _angle_settings = _AngleSettings(session, "angles-torsions")
        settings = _angle_settings
    else:
        raise ValueError("Settings type '%s' not implemented" % settings_type)
    return settings
