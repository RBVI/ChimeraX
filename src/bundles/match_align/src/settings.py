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

defaults = {
    'circular': False,
    'column_criterion': "any",
    'cutoff_distance': 5.0,
    'gap_char': '.',
    'iterate': False,
    'max_iterations': 3,
    'min_stretch': 3,
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _MatchAlignSettings(Settings):
    EXPLICIT_SAVE = deepcopy(defaults)

# for the GUI
_settings = None
def get_settings(session):
    global _settings
    # don't initialize a zillion times, which would also overwrite
    # any changed but not saved settings
    if _settings is None:
        # Windows doesn't allow '>' in file names, so change to an actual right arrow there; leave the same
        # on other platforms for backwards compatibility since this has been out in the wild for awhile
        import sys
        if sys.platform.startswith("win"):
            settings_name = "match\N{RIGHTWARDS ARROW}align"
        else:
            settings_name = "match->align"
        _settings = _MatchAlignSettings(session, settings_name)
    return _settings
