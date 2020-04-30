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


class ColoringTool(ToolInstance):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QDialogButtonBox, QWidget, QPushButton, QLabel
        from PyQt5.QtGui import QColor, QPixmap, QIcon
        from PyQt5.QtCore import Qt
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        main_dialog_area = QWidget()
        layout.addWidget(main_dialog_area)
        main_layout = QHBoxLayout()
        main_dialog_area.setLayout(main_layout)

        fav_color_area = QWidget()
        main_layout.addWidget(fav_color_area)
        fav_color_layout = QVBoxLayout()
        fav_color_layout.setContentsMargins(0,0,0,0)
        fav_color_layout.setSpacing(0)
        fav_color_area.setLayout(fav_color_layout)
        for spaced_name in [ "red", "orange red", "orange", "yellow", "lime", "forest green", "cyan",
                "light sea green", "blue", "cornflower blue", "medium blue", "purple", "hot pink",
                "magenta", "white", "light gray", "gray", "dark gray", "dim gray", "black"]:
            svg_name = "".join(spaced_name.split())
            color = QColor(svg_name)
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            button = QPushButton(icon, spaced_name.title())
            button.released.connect(lambda clr=spaced_name: self._color(clr))
            button.setStyleSheet("QPushButton { text-align: left; }")
            fav_color_layout.addWidget(button)

        actions_area = QWidget()
        main_layout.addWidget(actions_area)
        actions_layout = QVBoxLayout()
        actions_area.setLayout(actions_layout)
        header = QLabel("Coloring applies to:")
        header.setWordWrap(True)
        header.setAlignment(Qt.AlignCenter)
        actions_layout.addWidget(header)

        from PyQt5.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Close | qbbox.Help)
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda run=run, ses=self.session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def _color(self, color_name):
        print("color", color_name)
