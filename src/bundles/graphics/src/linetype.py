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

# LineType values match X3D specification.  And those match the
# "Linetype Section of the International Register of Graphical Items" 
# <http://www.cgmopen.org/technical/registry/>.

from enum import Enum


class LineType(Enum):
    Solid = 1
    Dashed = 2
    Dotted = 3
    DashedDotted = 4
    DashDotDot = 5
    CustomLine = 16
