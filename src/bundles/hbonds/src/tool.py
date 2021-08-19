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


class HBondsTool(ToolInstance):

    help = "help:user/tools/hbonds.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QDialogButtonBox
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        from .gui import HBondsGUI
        self.gui = HBondsGUI(session, show_model_restrict=False)
        layout.addWidget(self.gui)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.run_hbonds)
        bbox.button(qbbox.Apply).clicked.connect(self.run_hbonds)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        reset_button = bbox.addButton("Reset", qbbox.ActionRole)
        reset_button.setToolTip("Reset to initial-installation defaults")
        reset_button.clicked.connect(lambda *args: self.gui.reset())
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def run_hbonds(self):
        from chimerax.core.commands import run
        run(self.session, " ".join(self.gui.get_command()))
        self.session.logger.status("You can hide/close H-bonds with the Model Panel", secondary=True,
            color="blue", blank_after=15)
