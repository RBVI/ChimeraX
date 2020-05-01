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


class ColorActions(ToolInstance):

    SESSION_ENDURING = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area

        from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QDialogButtonBox, QWidget, QPushButton, \
            QLabel, QCheckBox, QFrame
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
        main_layout.addWidget(actions_area, alignment=Qt.AlignCenter)
        actions_layout = QVBoxLayout()
        actions_area.setLayout(actions_layout)
        header = QLabel("Coloring applies to:")
        header.setWordWrap(True)
        header.setAlignment(Qt.AlignCenter)
        actions_layout.addWidget(header, alignment=Qt.AlignBottom | Qt.AlignHCenter)
        self.target_button_info = []
        for label, target, initial_on in [("atoms/bonds", 'a', True),  ("cartoons", 'c', True),
                ("surfaces", 's', True), ("pseudobonds", 'p', True), ("ring fill", 'f', True),
                ("labels", 'l', False)]:
            chk = QCheckBox(label)
            chk.setChecked(initial_on)
            chk.clicked.connect(self._clear_global_buttons)
            actions_layout.addWidget(chk)
            self.target_button_info.append((chk, target))

        sep = QFrame()
        sep.setFrameStyle(QFrame.HLine)
        actions_layout.addWidget(sep, stretch=1)

        self.global_button_info = []
        for label, command in [("background", "set bg %s")]:
            chk = QCheckBox(label)
            chk.setChecked(False)
            chk.clicked.connect(self._clear_targeted_buttons)
            actions_layout.addWidget(chk)
            self.global_button_info.append((chk, command))

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

    @classmethod
    def get_singleton(cls, session, tool_name):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, tool_name)

    def _clear_targeted_buttons(self, *args):
        for button, *args in self.target_button_info:
            button.setChecked(False)

    def _clear_global_buttons(self, *args):
        for button, *args in self.global_button_info:
            button.setChecked(False)

    def _color(self, color_name):
        from chimerax.core.errors import UserError
        from chimerax.core.commands import run, StringArg
        target = ""
        for but, targ_char in self.target_button_info:
            if but.isChecked():
                target += targ_char
        commands = []
        if target:
            commands.append("color "
                + ("" if self.session.selection.empty() else "sel ")
                + StringArg.unparse(color_name)
                + ("" if target == "acspf" else " target " + target))

        for but, cmd in self.global_button_info:
            if but.isChecked():
                commands.append(cmd % StringArg.unparse(color_name))

        if commands:
            run(self.session, " ; ".join(commands))
        else:
            raise UserError("No target buttons for the coloring action are checked")
