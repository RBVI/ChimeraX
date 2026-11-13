# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
views = ["fourup", "sidebyside", "overunder"]
from chimerax.segmentations.view.modes import ViewMode


__all__ = ['views', 'FourPanelView', 'ViewMode']


def __getattr__(name):
    if name == 'FourPanelView':
        from .ui import FourPanelView
        globals()[name] = FourPanelView
        return FourPanelView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
