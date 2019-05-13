# vim: set expandtab ts=4 sw=4:

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

class AssociationsTool:

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window

        from PyQt5.QtWidgets import QHBoxLayout
        layout = QHBoxLayout()
        from chimerax.core.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(sv.session, selection_mode='single')
        self.chain_list.value_changed.connect(self._chain_changed)
        layout.addWidget(self.chain_list)
        tool_window.ui_area.setLayout(layout)

    def _chain_changed(self):
        if self.chain_list.count() == 0:
            self.tool_window.shown = False
