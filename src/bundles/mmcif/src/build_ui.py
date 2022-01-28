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

from Qt.QtWidgets import QVBoxLayout, QLabel, QGridLayout, QRadioButton, QLineEdit, QWidget, QHBoxLayout
from Qt.QtWidgets import QCheckBox, QSizePolicy
from Qt.QtCore import Qt
from chimerax.core.errors import UserError
from chimerax.build_structure import StartStructureProvider

class CCDProvider(StartStructureProvider):
    def command_string(self, widget):
        ccd_edit = widget.findChild(QLineEdit, "CCD")
        text = ccd_edit.text().strip()
        if not text:
            raise UserError("No CCD ID given")
        from chimerax.core.commands import StringArg
        return "open %s from ccd" % StringArg.unparse(text)

    def fill_parameters_widget(self, widget):
        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(QLabel("CCD ID:"), 1, 0, alignment=Qt.AlignRight)
        ccd_edit = QLineEdit()
        ccd_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(ccd_edit, 1, 1)
        ccd_edit.setObjectName("CCD")
        hint = QLabel('Input a <a href="https://www.wwpdb.org/data/ccd">PDB Chemical Component Dictionary</a> ID')
        hint.setOpenExternalLinks(True)
        layout.addWidget(hint, 2, 0, 1, 2, alignment=Qt.AlignCenter)
        layout.setRowStretch(3, 1)
