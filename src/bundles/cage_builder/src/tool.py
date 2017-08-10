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

# -----------------------------------------------------------------------------
# User interface for building cages.
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class CageBuilder(ToolInstance):

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self.minimize_steps = 10
        self.edge_thickness = 0.1  # Edge diameter as fraction of edge length.

        self.display_name = 'Cage Builder'

        from chimerax.core.ui.gui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QLineEdit
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        cp = QLabel('Create polygon')
        layout.addWidget(cp)
        b5 = QPushButton('5')
        b5.clicked.connect(lambda e: self.attach_polygons(5))
        layout.addWidget(b5)
        b6 = QPushButton('6')
        b6.clicked.connect(lambda e: self.attach_polygons(6))
        layout.addWidget(b6)
        mn = QPushButton('Minimize')
        mn.clicked.connect(self.minimize_cb)
        layout.addWidget(mn)
        ell = QLabel(' Edge length')
        layout.addWidget(ell)
        self.edge_length = el = QLineEdit('50')
        el.setMaximumWidth(30)
        layout.addWidget(el)
        layout.addStretch(1)    # Extra space at end of button row.
        parent.setLayout(layout)

        tw.manage(placement="side")

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def attach_polygons(self, n):
        d = self.vertex_degree()
        length, thickness, inset = self.edge_size()
        from . import cage
        cage.attach_polygons(self.session, 'selected', n, length, thickness, inset,
                             vertex_degree = d)

    def edge_size(self):
        el = self.edge_length
        elen = el.text()
        e = float(elen)
        et = self.edge_thickness*e    # Edge thickness
        ei = 0.5*et    # Edge inset
        return e, et, ei

    def vertex_degree(self):
        return 3
    
    def minimize_cb(self, event):
        from . import cage
        for i in range(self.minimize_steps):
            cage.optimize_shape(self.session)

def cage_builder_panel(session, tool_name):
  cb = getattr(session, '_cage_builder', None)
  if cb is None:
    session._cage_builder = cb = CageBuilder(session, tool_name)
  return cb
