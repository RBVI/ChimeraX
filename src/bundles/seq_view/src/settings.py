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

from chimerax.ui.options import Option, BooleanOption, EnumOption, IntOption, \
    OptionalRGBAOption, OptionalRGBAPairOption

APPEARANCE = "Appearance"
HEADERS = "Headers"
REGIONS = "Regions"

CSN_MAJ_GAP = "majority in column including gaps"
CSN_MAJ_NOGAP = "majority in column ignoring gaps"
class ConsensusStyleOption(EnumOption):
    values = (CSN_MAJ_GAP, CSN_MAJ_NOGAP)

LINE_WRAP_BALLOON = 'Only applies if wrapping is on.\n' \
            'Fit into window size and wrap at a multiple of the given value.'
defaults = {
    "block_space": (APPEARANCE,
        "Vertically separate wrapped blocks", 4, BooleanOption, {}, True),
    SINGLE_PREFIX + "block_space": (APPEARANCE,
        "Vertically separate wrapped lines", 4, BooleanOption, {}, False),
	"column_separation": (APPEARANCE,
        "Horizontal spacing (pixels)", 1, IntOption, {}, 0),
    SINGLE_PREFIX + "column_separation": (APPEARANCE,
        "Horizontal spacing (pixels)", 1, IntOption, {}, -2),
    ALIGNMENT_PREFIX + "consensus_style": (HEADERS, "Consensus style", 1,
        ConsensusStyleOption, {}, CSN_MAJ_GAP),
    "error_region_shown": (REGIONS,
        "Show structure-mismatch regions", 9, BooleanOption, {}, True),
    "error_region_borders": (REGIONS, "Structure-mismatch border", 10,
        OptionalRGBAPairOption, {'labels': ("full", "partial")}, (None, None)),
    "error_region_interiors": (REGIONS, "Structure-mismatch interior", 11,
        OptionalRGBAPairOption, {'labels': ("full", "partial")}, ((1.0, 0.3, 0.3, 1.0), "pink")),
    "gap_region_shown": (REGIONS, "Show missing-structure regions", 6, BooleanOption, {}, True),
    "gap_region_borders": (REGIONS, "Missing-structure border", 7, OptionalRGBAPairOption,
        {'labels': ("full", "partial")}, ("black", [chan/255.0 for chan in (190, 190, 190, 255)])),
    "gap_region_interiors": (REGIONS, "Missing-structure interior", 8,
        OptionalRGBAPairOption, {'labels': ("full", "partial")}, (None, None)),
    "line_width_multiple": (APPEARANCE, "Wrap lines at multiple of", 3, IntOption,
        {'balloon': LINE_WRAP_BALLOON, 'min': 1}, 5),
    SINGLE_PREFIX + "line_width_multiple": (APPEARANCE, "Wrap lines at multiple of", 3, IntOption,
        {'balloon': LINE_WRAP_BALLOON}, 5),
    "new_region_border": (REGIONS, "New-region border", 1, OptionalRGBAOption, {}, None),
    "new_region_interior": (REGIONS, "New-region interior", 2, OptionalRGBAOption, {},
        [chan/255.0 for chan in (233, 218, 198, 255)]),
    "sel_region_border": (REGIONS, "Selected-structure border", 4,
        OptionalRGBAOption, {}, None),
    "sel_region_interior": (REGIONS, "Selected-structure interior", 5,
        OptionalRGBAOption, {}, "light green"),
    ALIGNMENT_PREFIX + "show_ruler_at_startup": (APPEARANCE, "Show numbering at startup", 5,
        BooleanOption, {}, True),
    "show_sel": (REGIONS, "Show selection region", 3, BooleanOption, {}, True),
    SINGLE_PREFIX + "wrap": (APPEARANCE, "Wrap", 2, BooleanOption, {}, True),
    ALIGNMENT_PREFIX + "wrap_threshold":
        (APPEARANCE, "Wrap if", 2, IntOption, {'trailing_text': "sequences or fewer"}, 8),
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _SVSettings(Settings):
    EXPLICIT_SAVE = { k: v[-1] for k, v in defaults.items() }
    AUTO_SAVE = {
        "scf_colors_structures": True,
        "startup_headers": None,
    }

def init(session):
    # each SV instance has its own settings instance
    return _SVSettings(session, "Sequence Viewer")
