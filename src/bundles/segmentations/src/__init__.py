# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
__version__ = "3.5.11"

from chimerax.core import runtime_env_is_chimerax_app as _runtime_env_is_chimerax_app
from chimerax.segmentations.segmentation import Segmentation, open_grids_as_segmentation

__all__ = ["Segmentation", "open_grids_as_segmentation"]

if _runtime_env_is_chimerax_app():
    from chimerax.segmentations.bundle import SegmentationsBundle

    bundle_api = SegmentationsBundle()
    __all__.append("bundle_api")
