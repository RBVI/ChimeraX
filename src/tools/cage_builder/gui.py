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

    SIZE = (-1, -1)
    SESSION_SKIP = True

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)

        self.minimize_steps = 10
        self.edge_thickness = 0.1  # Edge diameter as fraction of edge length.

        self.display_name = 'Cage Builder'

        from chimerax.core import window_sys
        kw = {'size': self.SIZE} if window_sys == 'wx' else {}
        from chimerax.core.ui.gui import MainToolWindow
        tw = MainToolWindow(self, **kw)
        self.tool_window = tw
        parent = tw.ui_area

        if window_sys == 'wx':
            import wx
            bl = wx.StaticText(parent, label="Create polygon")

            b5 = wx.Button(parent, label='5', style=wx.BU_EXACTFIT)
            b5.Bind(wx.EVT_BUTTON, lambda e: self.attach_polygons(5))
            b6 = wx.Button(parent, label='6', style=wx.BU_EXACTFIT)
            b6.Bind(wx.EVT_BUTTON, lambda e: self.attach_polygons(6))
            bm = wx.Button(parent, label='Minimize', style=wx.BU_EXACTFIT)
            bm.Bind(wx.EVT_BUTTON, self.minimize_cb)
            bel = wx.StaticText(parent, label="Edge length")
            self.edge_length = be = wx.TextCtrl(parent, value = '50', size = (30,-1))

            sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(bl, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
            sizer.Add(b5, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
            sizer.Add(b6, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
            sizer.Add(bm, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
            sizer.Add(bel, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
            sizer.Add(be, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
            parent.SetSizerAndFit(sizer)
        elif window_sys == 'qt':
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
            layout.addStretch(1)	# Extra space at end of button row.
            parent.setLayout(layout)

        tw.manage(placement="right")

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
        from chimerax.core import window_sys
        if window_sys == 'wx':
            elen = el.GetValue()
        elif window_sys == 'qt':
            elen = el.text()
        e = float(elen)
        et = self.edge_thickness*e	# Edge thickness
        ei = 0.5*et	# Edge inset
        return e, et, ei

    def vertex_degree(self):
        return 3
    
    def minimize_cb(self, event):
        from . import cage
        for i in range(self.minimize_steps):
            cage.optimize_shape(self.session)

def cage_builder_panel(session, bundle_info):
  cb = getattr(session, '_cage_builder', None)
  if cb is None:
    session._cage_builder = cb = CageBuilder(session, bundle_info)
  return cb
