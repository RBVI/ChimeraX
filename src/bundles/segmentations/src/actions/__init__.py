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
from enum import IntEnum
from chimerax.core.commands import run


class ImageFormat(IntEnum):
    DICOM = 0
    NIFTI = 1
    NRRD = 2

    def __str__(self):
        if self.name == "NIFTI":
            return "NIfTI"
        return self.name


class MouseAction(IntEnum):
    NONE = 0
    ADD_TO_SEGMENTATION = 1
    MOVE_SPHERE = 2
    RESIZE_SPHERE = 3
    ERASE_FROM_SEGMENTATION = 4

    def __str__(self):
        return " ".join(self.name.split("_")).lower()


class HandAction(IntEnum):
    NONE = 0
    RESIZE_CURSOR = 1
    MOVE_CURSOR = 2
    ADD_TO_SEGMENTATION = 3
    ERASE_FROM_SEGMENTATION = 4

    def __str__(self):
        return " ".join(self.name.split("_")).lower()


class Handedness(IntEnum):
    LEFT = 0
    RIGHT = 1

    def __str__(self):
        return self.name.title().lower()


def run_toolbar_button(session, name):
    # run shortcut chosen via bundle provider interface
    from chimerax.segmentations.ui.segmentation_mouse_mode import (
        mouse_bindings_saved,
        hand_bindings_saved,
    )
    if name == "toggle mouse modes":
        if mouse_bindings_saved():
            run(session, "segmentations mouseModes off")
        else:
            run(session, "segmentations mouseModes on")
    elif name == "toggle hand modes":
        if hand_bindings_saved():
            run(session, "segmentations handModes off")
        else:
            run(session, "segmentations handModes on")
    else:
        raise ValueError("No provider for toolbar button %s" % name)
