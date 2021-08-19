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
from Qt.QtGui import QDoubleValidator, QIntValidator
from Qt.QtCore import Qt
from chimerax.core.errors import UserError
from chimerax.build_structure import StartStructureProvider

class PubChemProvider(StartStructureProvider):
    def command_string(self, widget):
        cid_edit = widget.findChild(QLineEdit, "CID")
        text = cid_edit.text().strip()
        if not text:
            raise UserError("No PubChem CID given")
        return "open %s from pubchem" % text

    def fill_parameters_widget(self, widget):
        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.setRowStretch(0, 1)
        layout.addWidget(QLabel("PubChem CID:"), 1, 0, alignment=Qt.AlignRight)
        cid_edit = QLineEdit()
        cid_validator = QIntValidator()
        cid_validator.setBottom(1)
        cid_edit.setValidator(cid_validator)
        layout.addWidget(cid_edit, 1, 1, alignment=Qt.AlignLeft)
        cid_edit.setObjectName("CID")
        acknowledgement = QLabel('PubChem CID support courtesy of <a href="https://pubchemdocs.ncbi.nlm.nih.gov/power-user-gateway">PubChem Power User Gateway (PUG) web services</a>')
        acknowledgement.setWordWrap(True)
        acknowledgement.setOpenExternalLinks(True)
        layout.addWidget(acknowledgement, 2, 0, 1, 2)
        layout.setRowStretch(3, 1)
