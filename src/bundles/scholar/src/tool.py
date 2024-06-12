# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow


class ScholARTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    help = "help:user/tools/scholar.html"

    def __init__(self, session):
        self.display_name = "Schol-AR"
        super().__init__(session, self.display_name)
        self._build_ui()

    def _build_ui(self):
        """
        Build the Schol-AR Qt GUI.
        """
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
