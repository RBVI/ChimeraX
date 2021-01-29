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
    help = "help:user/tools/cagebuilder.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self.minimize_steps = 10
        self.edge_thickness = 0.1  # Edge diameter as fraction of edge length.

        self.display_name = 'Cage Builder'

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from Qt.QtWidgets import QHBoxLayout, QLabel, QPushButton, QLineEdit
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
        dp = QPushButton('Delete')
        dp.clicked.connect(self.delete_polygon_cb)
        layout.addWidget(dp)
        layout.addStretch(1)    # Extra space at end of button row.
        parent.setLayout(layout)

        tw.manage(placement="side")

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, CageBuilder, 'Cage Builder', create=create)

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
    
    def delete_polygon_cb(self, event):
        from . import cage
        polys = cage.selected_polygons(self.session)
        cage.delete_polygons(polys)
