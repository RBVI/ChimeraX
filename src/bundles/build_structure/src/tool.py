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
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QMenu, QStackedWidget, QWidget, QLabel, QFrame
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtCore import Qt

class BuildStructureTool(ToolInstance):

    #help = "help:user/tools/hbonds.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(3)
        parent.setLayout(layout)

        self.category_button = QPushButton()
        layout.addWidget(self.category_button, alignment=Qt.AlignCenter)
        cat_menu = QMenu()
        self.category_button.setMenu(cat_menu)
        cat_menu.triggered.connect(self._cat_menu_cb)

        self.category_areas = QStackedWidget()
        layout.addWidget(self.category_areas)

        self.category_widgets = {}
        for category in ["Modify Structure"]:
            self.category_widgets[category] = widget = QFrame()
            widget.setLineWidth(2)
            widget.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            getattr(self, "_layout_" + category.lower().replace(' ', '_'))(widget)
            self.category_areas.addWidget(widget)
        self.category_button.setText(category)
        self.category_areas.setCurrentWidget(widget)

        tw.manage(placement="side")

    def run_cmd(self, cmd):
        from chimerax.core.commands import run
        run(self.session, " ".join(cmd))

    def _cat_menu_cb(self, action):
        self.category_areas.setCurrentWidget(self.category_widgets[action.text()])

    def _layout_modify_structure(self, parent):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        layout.addWidget(QLabel("Change selected atoms to..."), alignment=Qt.AlignHCenter | Qt.AlignBottom)
        frame = QFrame()
        layout.addWidget(frame, alignment=Qt.AlignHCenter | Qt.AlignTop)
        frame.setLineWidth(1)
        frame.setFrameStyle(QFrame.Panel | QFrame.Plain)
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(0,0,0,0)
        frame_layout.setSpacing(0)
        frame.setLayout(frame_layout)
        params_layout = QGridLayout()
        params_layout.setSpacing(10)
        frame_layout.addLayout(params_layout)
        for col, title in enumerate(["Element", "Bonds", "Geometry"]):
            params_layout.addWidget(QLabel(title), 0, col)
