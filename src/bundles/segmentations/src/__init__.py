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
__version__ = "3.5.12"

from chimerax.core import runtime_env_is_chimerax_app as _runtime_env_is_chimerax_app

__all__ = ["Segmentation", "open_grids_as_segmentation"]


def __getattr__(name):
    if name in ("Segmentation", "open_grids_as_segmentation"):
        from importlib import import_module
        value = getattr(import_module('.segmentation', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


if _runtime_env_is_chimerax_app():
    from chimerax.segmentations.bundle import SegmentationsBundle

    bundle_api = SegmentationsBundle()
    __all__.append("bundle_api")
