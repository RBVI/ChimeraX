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

class ViewMode(IntEnum):
    TWO_BY_TWO = 0
    ORTHOPLANES_OVER_3D = 1
    ORTHOPLANES_BESIDE_3D = 2
    DEFAULT_DESKTOP = 3
    DEFAULT_VR = 4

    def __str__(self):
        if self.name == "TWO_BY_TWO":
            return "2 x 2 (desktop)"
        elif self.name == "ORTHOPLANES_OVER_3D":
            return "3D over slices (desktop)"
        elif self.name == "ORTHOPLANES_BESIDE_3D":
            return "3D beside slices (desktop)"
        elif self.name == "DEFAULT_DESKTOP":
            return "3D only (desktop)"
        elif self.name == "DEFAULT_VR":
            return "3D only (VR)"
        return "%s: Set a value to return for the name of this EnumItem" % self.name
