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

from chimerax.core.tools import ToolInstance

class FilePanel(ToolInstance):

    SESSION_ENDURING = True
    help = "help:user/tools/filehistory.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys = False)
        parent = self.tool_window.ui_area

        from chimerax.core.ui.file_history import FileHistory
        fh = FileHistory(session, parent, size_hint=(575, 200))
        self.tool_window.manage(placement="side")

