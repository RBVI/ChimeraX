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

# ToolUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
#
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance


class ToolUI(ToolInstance):

    SESSION_ENDURING = False
    # if SESSION_ENDURING is True, tool instance not deleted at session closure

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)
        # 'display_name' defaults to class name with spaces inserted
        # between lower-then-upper-case characters (therefore "Tool UI"
        # in this case), so only override if different name desired
        self.display_name = "custom name for running tool"
        if session.ui.is_gui:
            from chimerax.core.ui.gui import MainToolWindow
            self.tool_window = MainToolWindow(self)
            self.tool_window.manage(placement="side")
            parent = self.tool_window.ui_area
            # TODO: UI content code goes here
