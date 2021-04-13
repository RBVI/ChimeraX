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

        models_layout = QVBoxLayout()
        layout.addLayout(models_layout, stretch=1)
        models_layout.setSpacing(0)
        models_label = QLabel("Save models")
        from chimerax.ui import shrink_font
        shrink_font(models_label)
        models_layout.addWidget(models_label, alignment=Qt.AlignLeft)
        from chimerax.atomic.widgets import StructureListWidget
        self.structure_list = StructureListWidget(session)
        self.structure_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        models_layout.addWidget(self.structure_list)

        self.setLayout(layout)

    def options_string(self):
        models = self.structure_list.value
        from chimerax.core.errors import UserError
        if not models:
            raise UserError("No models chosen for saving")
        from chimerax.atomic import Structure
        from chimerax.core.commands import concise_model_spec
        spec = concise_model_spec(self.session, models, relevant_types=Structure)
        if spec:
            cmd = "models " + spec
        else:
            cmd = ""
        return cmd
