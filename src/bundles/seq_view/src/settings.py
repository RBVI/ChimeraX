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

from chimerax.core.options import Option, BooleanOption, IntOption, OptionalRGBAOption
class OptionalRGBAPair(Option): pass # maybe make it a real option?

defaults = {
    "block space": (BooleanOption, True),
    SINGLE_PREFIX + "block space": (BooleanOption, False),
	"column_separation": (IntOption, 0),
    SINGLE_PREFIX + "column_separation": (IntOption, -2),
    "error_region_shown": (BooleanOption, True),
    "error_region_borders": (OptionalRGBAPair, (None, None)),
    "error_region_interiors": (OptionalRGBAPair, ((1.0, 0.3, 0.3, 1.0), "pink")),
    "gap_region_shown": (BooleanOption, True),
    "gap_region_borders": (OptionalRGBAPair, ("black",
        [chan/255.0 for chan in (190, 190, 190, 255)])),
    "gap_region_interiors": (OptionalRGBAPair, (None, None)),
    "line_width": (IntOption, -5),
    SINGLE_PREFIX + "line_width": (IntOption, -5),
    "new_region_border": (OptionalRGBAOption, None),
    "new_region_interior": (OptionalRGBAOption, [chan/255.0 for chan in (233, 218, 198, 255)]),
    "sel_region_border": (OptionalRGBAOption, None),
    "sel_region_interior": (OptionalRGBAOption, "light green"),
    "show_ruler_at_startup": (BooleanOption, True),
    "show_sel": (BooleanOption, True),
    # if 'wrap_if' is True, then alignment will be wrapped if the number of
    # sequences is no more than 'wrap_threshold'.  If 'wrap_if' is false, then
    # wrapping will occur if 'wrap' is true.  Wrapping will occur at
    # 'line_width' columns, if positive.  if negative, alignment will be
    # wrapped to window size, at a multiple of abs(line_width) characters.
    "wrap": (BooleanOption, False),
    SINGLE_PREFIX + "wrap": (BooleanOption, True),
    "wrap_if": (BooleanOption, True),
    SINGLE_PREFIX + "wrap_if": (BooleanOption, False),
    "wrap_threshold": (IntOption, 8),
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _SVSettings(Settings):
	EXPLICIT_SAVE = { k: v[-1] for k, v in defaults.items() }

def init(session):
    # each SV instance has its own settings instance
    return _SVSettings(session, "Sequence Viewer")
