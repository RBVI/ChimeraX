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

class SmilesProvider(StartStructureProvider):
    def command_string(self, widget):
        smiles_edit = widget.findChild(QLineEdit, "SMILES")
        text = smiles_edit.text().strip()
        if not text:
            raise UserError("No SMILES given")
        from chimerax.core.commands import StringArg
        return "open %s from smiles" % StringArg.unparse(text)

    def fill_parameters_widget(self, widget):
        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,5)
        layout.setSpacing(0)
        widget.setLayout(layout)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(QLabel("SMILES string:"), 1, 0, alignment=Qt.AlignRight)
        smiles_edit = QLineEdit()
        smiles_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(smiles_edit, 1, 1)
        smiles_edit.setObjectName("SMILES")
        from .smiles import fetcher_info
        providers = " or ".join(['<a href="%s">%s</a>' % (info[3], info[2]) for info in fetcher_info])
        acknowledgement = QLabel('SMILES support courtesy of %s' % providers)
        #acknowledgement.setWordWrap(True) -- perhaps the HTML confuses it?
        acknowledgement.setOpenExternalLinks(True)
        layout.addWidget(acknowledgement, 2, 0, 1, 2, alignment=Qt.AlignCenter)
        layout.setRowStretch(3, 1)
