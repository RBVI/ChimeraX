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
ALIGNMENT_PREFIX = "alignment_"

from chimerax.core.ui.options import Option, BooleanOption, IntOption, OptionalRGBAOption, \
    OptionalRGBAPairOption

APPEARANCE = "Appearance"
REGIONS = "Regions"

LINE_WRAP_BALLOON = 'Only applies if wrapping is on.  If positive, wrap lines at that length.\n' \
            'If negative, fit into window size and wrap at a multiple of the given value.'
defaults = {
    "block_space": (APPEARANCE,
        "Put vertical space between wrapped lines of sequence", BooleanOption, {}, True),
    SINGLE_PREFIX + "block_space": (APPEARANCE,
        "Put vertical space between wrapped blocks of sequences", BooleanOption, {}, False),
	"column_separation": (APPEARANCE,
        "Separation between columns, in pixels", IntOption, {}, 0),
    SINGLE_PREFIX + "column_separation": (APPEARANCE,
        "Separation between columns, in pixels", IntOption, {}, -2),
    "error_region_shown": (REGIONS,
        "Show structure mismatches", BooleanOption, {}, True),
    "error_region_borders": (REGIONS, "Full/partial structure-mismatch border colors",
        OptionalRGBAPairOption, {}, (None, None)),
    "error_region_interiors": (REGIONS, "Full/partial structure-mismatch interior colors",
        OptionalRGBAPairOption, {}, ((1.0, 0.3, 0.3, 1.0), "pink")),
    "gap_region_shown": (REGIONS, "Show missing structure", BooleanOption, {}, True),
    "gap_region_borders": (REGIONS, "Full/partial missing-structure border colors",
        OptionalRGBAPairOption, {}, ("black", [chan/255.0 for chan in (190, 190, 190, 255)])),
    "gap_region_interiors": (REGIONS, "Full/partial missing-structure interior colors",
        OptionalRGBAPairOption, {}, (None, None)),
    "line_width": (APPEARANCE, "Line-wrapping length", IntOption,
        {'balloon': LINE_WRAP_BALLOON}, -5),
    SINGLE_PREFIX + "line_width": (APPEARANCE, "Line-wrapping length", IntOption,
        {'balloon': LINE_WRAP_BALLOON}, -5),
    "new_region_border": (REGIONS, "New region border color, if any", OptionalRGBAOption, {}, None),
    "new_region_interior": (REGIONS, "New region interior color, if any", OptionalRGBAOption, {},
        [chan/255.0 for chan in (233, 218, 198, 255)]),
    "sel_region_border": (REGIONS, "Selected structure border color, if any",
        OptionalRGBAOption, {}, None),
    "sel_region_interior": (REGIONS, "Selected structure interior color, if any",
        OptionalRGBAOption, {}, "light green"),
    ALIGNMENT_PREFIX + "show_ruler_at_startup": (APPEARANCE, "Show numbering at startup",
        BooleanOption, {}, True),
    "show_sel": (REGIONS, "Show selection", BooleanOption, {}, True),
    # if 'wrap_if' is True, then alignment will be wrapped if the number of
    # sequences is no more than 'wrap_threshold'.  If 'wrap_if' is false, then
    # wrapping will occur if 'wrap' is true.  Wrapping will occur at
    # 'line_width' columns, if positive.  if negative, alignment will be
    # wrapped to window size, at a multiple of abs(line_width) characters.
    "wrap": (APPEARANCE, "Wrap sequences (if not based on number of sequences)",
        BooleanOption, {}, False),
    SINGLE_PREFIX + "wrap": (APPEARANCE, "Wrap sequences (if not based on number of sequences)",
        BooleanOption, {}, True),
    "wrap_if":
        (APPEARANCE, "Whether to wrap based on number of sequences", BooleanOption, {}, True),
    SINGLE_PREFIX + "wrap_if":
        (APPEARANCE, "Whether to wrap based on number of sequences", BooleanOption, {}, False),
    ALIGNMENT_PREFIX + "wrap_threshold":
        (APPEARANCE, "Wrap if this number of sequences or less", IntOption, {}, 8),
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _SVSettings(Settings):
	EXPLICIT_SAVE = { k: v[-1] for k, v in defaults.items() }

def init(session):
    # each SV instance has its own settings instance
    return _SVSettings(session, "Sequence Viewer")
