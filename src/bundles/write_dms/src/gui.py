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

from Qt.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QPushButton, QMenu, \
    QGridLayout, QSizePolicy
from Qt.QtCore import Qt

class SaveOptionsWidget(QFrame):
    def __init__(self, session):
        super().__init__()
        self.session = session

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(5)

        surface_layout = QHBoxLayout()
        surface_layout.setSpacing(0)
        layout.addLayout(surface_layout)
        surface_layout.addStretch(1)
        surface_layout.addWidget(QLabel("Save surface "))
        from chimerax.atomic import MolecularSurface
        from chimerax.ui.widgets import ModelMenuButton
        self.surface_menu = ModelMenuButton(session, class_filter=MolecularSurface,
            no_value_button_text="No surface chosen")
        surface_layout.addWidget(self.surface_menu)
        surface_layout.addStretch(1)

        from .settings import get_settings
        self.settings = get_settings(session)

        normals_layout = QHBoxLayout()
        normals_layout.setSpacing(0)
        layout.addLayout(normals_layout)
        normals_layout.addStretch(1)
        self.normals_checkbox = QCheckBox("Save normals")
        self.normals_checkbox.setChecked(self.settings.save_normals)
        normals_layout.addWidget(self.normals_checkbox)
        normals_layout.addStretch(1)

        displayed_layout = QHBoxLayout()
        displayed_layout.setSpacing(0)
        layout.addLayout(displayed_layout)
        displayed_layout.addStretch(1)
        self.displayed_checkbox = QCheckBox("Limit output to displayed surface sections")
        self.displayed_checkbox.setChecked(self.settings.displayed_only)
        displayed_layout.addWidget(self.displayed_checkbox)
        displayed_layout.addStretch(1)

        self.setLayout(layout)

    def options_string(self):
        surface = self.surface_menu.value
        total_surfaces = len(self.surface_menu.all_values)
        from chimerax.core.errors import UserError
        if not surface:
            if total_surfaces == 0:
                raise UserError("No molecular surfaces open")
            else:
                raise UserError("No surface chosen for saving")
        args = []
        if total_surfaces > 1:
            args.extend(["surface", surface.atomspec])

        disp_val = self.settings.displayed_only = self.displayed_checkbox.isChecked()
        normals_val = self.settings.save_normals = self.normals_checkbox.isChecked()
        from .settings import defaults
        if disp_val != defaults['displayed_only']:
            args.extend(["displayedOnly", str(disp_val).lower()])
        if normals_val != defaults['save_normals']:
            args.extend(["saveNormals", str(normals_val).lower()])
        return ' '.join(args)
