UC Medical Plan Satisfaction Survey
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

SINGLE_PREFIX = "single_seq_"

defaults = {
	"column_separation": 0,
    SINGLE_PREFIX + "column_separation": -2,
    "line_width": -10,
    SINGLE_PREFIX + "line_width": -10,
    "new_region_border": None,
    "new_region_interior": [chan/255.0 for chan in (233, 218, 198, 255)],
    "sel_region_border": None,
    "sel_region_interior": "light green",
    "show_sel": True,
    # if 'wrap_if' is True, then alignment will be wrapped if the number of
    # sequences is at least 'wrap_threshold'.  If 'wrap_if' is false, then
    # wrapping will occur is 'wrap' is true.  Wrapping will occur at
    # 'line_width' columns, if positive.  if negative, alignment will be
    # wrapped to window size, at a multiple of abs(line_width) characters.
    "wrap": False,
    SINGLE_PREFIX + "wrap": True,
    "wrap_if": True,
    SINGLE_PREFIX + "wrap_if": False,
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _MAVSettings(Settings):
	EXPLICIT_SAVE = deepcopy(defaults)

def init(session):
    # each MAV instance has its own settings instance
    return _MAVSettings(session, "Multalign Viewer")
