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
class MarkerModeSettings(ToolInstance):
    SESSION_ENDURING = True

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self.display_name = 'Marker Placement'

        self.mode_menu_names = mnames = {
            'maximum': 'Place marker at density maximum',
            'plane': 'Place marker on volume plane',
            'surface': 'Place marker on surface',
            'center': 'Place marker at center of connected surface',
            'point': 'Place marker at 3d pointer position',
            'link': 'Link consecutively clicked markers',
            'move': 'Move markers',
            'resize': 'Resize markers or links',
            'delete': 'Delete markers or links',
        }
        self.mode_order = ('maximum', 'plane', 'surface', 'center', 'point',
                          'link', 'move', 'resize', 'delete')

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys=False)
        self.tool_window = tw
        parent = tw.ui_area
        
        from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMenu, QSizePolicy, QCheckBox

        playout = QVBoxLayout(parent)
        playout.setContentsMargins(0,0,0,0)
        playout.setSpacing(0)
        parent.setLayout(playout)

        f = QFrame(parent)
        f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        playout.addWidget(f)
        layout = QVBoxLayout(f)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        f.setLayout(layout)

        # Toolbar icons for marker modes
        tb = self.create_buttons(parent)
        layout.addWidget(tb)

        # Option menu for marker modes
        mf = QFrame(f)
        mf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(mf)
        mm_layout = QHBoxLayout(mf)
        mm_layout.setContentsMargins(0,0,0,0)
        mm_layout.setSpacing(5)
        mf.setLayout(mm_layout)
        ml = QLabel(' Mouse mode', mf)
        mm_layout.addWidget(ml)
        self.mode_button = mb = QPushButton(mf)
        mm = QMenu()
        for m in self.mode_order:
            mm.addAction(mnames[m], lambda mode=m, self=self: self.mode_change_cb(mode))
        mb.setMenu(mm)
        mm_layout.addWidget(mb)
        mm_layout.addStretch(1)    # Extra space at end

        # Link consecutive markers checkbutton
        self.link_new_button = lc = QCheckBox('Link new marker to selected marker', f)
        lc.stateChanged.connect(self.link_new_cb)
        layout.addWidget(lc)

        self.update_settings()
        
        tw.manage(placement="side")

    def create_buttons(self, parent):
        from PyQt5.QtWidgets import QAction, QFrame, QHBoxLayout, QToolButton
        from PyQt5.QtGui import QIcon
        from PyQt5.QtCore import Qt, QSize
        tb = QFrame(parent)
        layout = QHBoxLayout(tb)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tb.setStyleSheet('QFrame{spacing:0px;}\n'
                         'QToolButton{padding:0px; margin:0px; border:none;}')
        for mname in self.mode_order:
            mdesc = self.mode_menu_names[mname]
            b = QToolButton(tb)
            b.setIconSize(QSize(40,40))
            from os import path
            icon_dir = path.join(path.dirname(__file__), 'icons')
            icon_path = path.join(icon_dir, mname + '.png')
            action = QAction(QIcon(icon_path), mdesc, tb)
            b.setDefaultAction(action)
            def button_press_cb(event, mode=mname, self=self):
                self.mode_change_cb(mode)
            action.triggered.connect(button_press_cb)
            layout.addWidget(b)
        layout.addStretch(1)    # Extra space at end
        return tb

    def update_settings(self):
        s = self.session
        from . import mouse
        mode = mouse._mouse_marker_settings(s, 'placement_mode')
        self.mode_button.setText(self.mode_menu_names[mode])
        lnew = mouse._mouse_marker_settings(s, 'link_new_markers')
        from PyQt5.QtCore import Qt
        self.link_new_button.setChecked(Qt.Checked if lnew else Qt.Unchecked)
        
    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def mode_change_cb(self, mode):
        self.mode_button.setText(self.mode_menu_names[mode])

        from . import mouse
        s = mouse._mouse_marker_settings(self.session)
        s['placement_mode'] = mode

    def link_new_cb(self, link):
        from . import mouse
        s = mouse._mouse_marker_settings(self.session)
        s['link_new_markers'] = link

        
def marker_panel(session, tool_name):
  cb = getattr(session, '_markers_gui', None)
  if cb is None:
    session._markers_gui = cb = MarkerModeSettings(session, tool_name)
  return cb
