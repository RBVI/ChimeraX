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

# -----------------------------------------------------------------------------
# Handle Horos medical imaging 3d preset .plist files and color lookup table
# .plist files for setting 3d image rendering colors, brightness and transparency.
#

from .colortables import appearance_names, AppearanceArg, appearance_settings
from .colortables import add_appearance, delete_appearance
