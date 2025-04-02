# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance
from chimerax.core.settings import Settings
from chimerax.core.configfile import Value
from Qt.QtWidgets import QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QButtonGroup, QRadioButton, QWidget
from Qt.QtWidgets import QPushButton, QScrollArea, QMenu, QCheckBox
from Qt.QtCore import Qt
from chimerax.core.commands import run
from chimerax.ui import tool_user_error
import json

class AnisoSettings(Settings):
    AUTO_SAVE = {
        "custom_presets": Value({}, json.loads, json.dumps),
    }

'''
        "simple ellipsoid": {
        "principal axes": {
        "principal ellipses": {
        "ellipsoid and principal axes": {
        "octant lines": {
        "snow globe axes": {
        "snow globe ellipses": {
'''

class AnisoTool(ToolInstance):

    #help = "help:user/tools/thermalellipsoids.html"

    NO_PRESET_TEXT = "no preset"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        parent.setLayout(main_layout)


        preset_model_layout = QHBoxLayout()
        main_layout.addLayout(preset_model_layout)
        preset_model_layout.addStretch(1)
        preset_model_layout.addWidget(QLabel("Preset:"), alignment=Qt.AlignRight)

        self.preset_menu_button = pmb = QPushButton()
        preset_menu = QMenu(pmb)
        pmb.setMenu(preset_menu)
        pmb.setText(self.NO_PRESET_TEXT)
        self._populate_preset_menu()
        preset_model_layout.addWidget(pmb, alignment=Qt.AlignLeft)
        preset_model_layout.addStretch(1)

        from chimerax.atomic.widgets import AtomicStructureMenuButton as ASMB
        self.structure_button = sb = ASMB(session, no_value_button_text="No relevant structures",
            filter_func=lambda s: s.atoms.has_aniso_u.any())
        preset_model_layout.addWidget(sb)
        preset_model_layout.addStretch(1)

        hide_show_layout = QHBoxLayout()
        main_layout.addLayout(hide_show_layout)
        hide_show_layout.addStretch(1)
        show_button = QPushButton("Show")
        show_button.clicked.connect(lambda *args, f=self._show_hide_cb: f("aniso"))
        hide_show_layout.addWidget(show_button)
        hide_show_layout.addWidget(QLabel("/"))
        hide_button = QPushButton("Hide")
        hide_button.clicked.connect(lambda *args, f=self._show_hide_cb: f("aniso hide"))
        hide_show_layout.addWidget(hide_button)
        hide_show_layout.addWidget(QLabel("depictions"))
        hide_show_layout.addStretch(1)
        sel_restrict_layout = QHBoxLayout()
        main_layout.addLayout(sel_restrict_layout)
        sel_restrict_layout.addStretch(1)
        self.sel_restrict_check_box = QCheckBox("Restrict Show/Hide to current selection, if any")
        sel_restrict_layout.addWidget(self.sel_restrict_check_box)
        sel_restrict_layout.addStretch(1)

        tw.manage(placement=None)

    def _populate_preset_menu(self):
        menu = self.preset_menu_button.menu()
        menu.clear()
        #TODO

    def _show_hide_cb(self, cmd):
        s = self.structure_button.value
        if not s:
            return tool_user_error("No structure chosen")

        spec = s.atomspec

        if self.sel_restrict_check_box.isChecked():
            spec += " & sel"

        run(self.session, cmd + ' ' + spec)
