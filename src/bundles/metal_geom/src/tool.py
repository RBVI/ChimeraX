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
'''
from chimerax.core.settings import Settings
from Qt.QtWidgets import QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QButtonGroup, QRadioButton, QWidget
from Qt.QtWidgets import QPushButton, QScrollArea, QMenu, QCheckBox, QLineEdit, QSpacerItem, QSizePolicy
from Qt.QtWidgets import QGroupBox, QInputDialog
from Qt.QtGui import QDoubleValidator, QIntValidator
from Qt.QtCore import Qt
from chimerax.core.commands import run
from chimerax.ui import tool_user_error
from chimerax.ui.widgets import ColorButton
from .cmd import builtin_presets

style_attrs = list(builtin_presets["simple"].keys())
'''

class MetalGeomTool(ToolInstance):

    #help = "help:user/tools/thermalellipsoids.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        '''
        from .settings import get_settings
        self.settings = get_settings(session)
        '''

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, statusbar=True)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton

        layout = QHBoxLayout()
        layout.setSpacing(2)
        parent.setLayout(layout)

        layout.addStretch(1)

        tables_layout = QVBoxLayout()
        layout.addLayout(tables_layout)

        tables_layout.addStretch(1)

        metal_buttons_layout = QHBoxLayout()
        tables_layout.addLayout(metal_buttons_layout)
        metal_buttons_layout.addStretch(1)
        from chimerax.atomic.widgets import AtomMenuButton
        def list_metals(ses=session):
            from chimerax.atomic import all_atoms
            atoms = all_atoms(session)
            return atoms.filter(atoms.elements.is_metal)
        self.metal_menu = AtomMenuButton(session, list_func=list_metals, add_numbering=True,
            autoselect=AtomMenuButton.AUTOSELECT_FIRST)
        self.metal_menu.value_changed.connect(self._populate_table)
        metal_buttons_layout.addWidget(self.metal_menu)
        self.next_metal_button = QPushButton("Next metal")
        self.next_metal_button.clicked.connect(self._next_metal)
        metal_buttons_layout.addWidget(self.next_metal_button)
        metal_buttons_layout.addStretch(1)

        tables_layout.addStretch(1)

        layout.addStretch(1)

        controls_layout = QVBoxLayout()
        layout.addLayout(controls_layout)

        controls_layout.addStretch(1)

        controls_layout.addWidget(QLabel("Metal transparency"))
        from chimerax.ui.widgets import FloatSlider
        self.transparency_slider = FloatSlider(0.0, 1.0, 0.01, 2, self._transparency_cb, display_value=False)
        self.transparency_slider.set_left_text("opaque")
        self.transparency_slider.set_right_text("transparent")
        controls_layout.addWidget(self.transparency_slider)

        controls_layout.addStretch(1)

        layout.addStretch(1)

        tw.manage(placement=None)

    def delete(self):
        '''
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()
        self.structure_button.destroy()
        '''
        super().delete()

    def _focus(self, focus_atoms):
        if focus_atoms:
            from chimerax.core.commands import run
            from chimerax.atomic import concise_atom_spec
            run(self.session, "view " + concise_atom_spec(self.session, focus_atoms))

    def _next_metal(self):
        metal = self.metal_menu.value
        metals = self.metal_menu.all_values
        self.metal_menu.value = metals[metals.index(metal)+1]

    def _populate_table(self):
        metal = self.metal_menu.value

        self.tool_window.status("")

        metals = self.metal_menu.all_values
        #TODO: update/invalidate gde_cache

        # hide/show "Next metal" button
        if len(metals) > 1:
            self.next_metal_button.setHidden(False)
            self.next_metal_button.setEnabled(metal != metals[-1])
        else:
            self.next_metal_button.setHidden(True)

        #TODO: handle transparency for new and old metal

        if not metal:
            #TODO: set widget states for no metal
            return

        #TODO: compute geometry

        #TODO: focus view based on geom data
        self._focus([metal]) # kludge so that the menu does something while implementation fleshed out

    def _transparency_cb(self, *args):
        print("_transparency_cb:", args)
