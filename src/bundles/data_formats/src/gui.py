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

from Qt.QtWidgets import QDialog, QVBoxLayout, QListWidget, QLabel, QCheckBox
from Qt.QtCore import Qt

class AskFormatDialog(QDialog):

    def __init__(self, session, suffix, formats):
        super().__init__()
        self.formats = formats
        self.chosen_format = None

        self.setWindowTitle("Choose Format")
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("Multiple formats support the %s suffix.\nPlease choose one." % suffix))
        self.list = QListWidget()
        self.list.addItems([fmt.name for fmt in formats])
        self.list.setSelectionMode(QListWidget.SingleSelection)
        self.list.itemSelectionChanged.connect(self.selection_changed)
        layout.addWidget(self.list)
        self.make_default = QCheckBox("Use chosen format in the future without asking")
        layout.addWidget(self.make_default)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Cancel)
        bbox.accepted.connect(self.choose_format)
        bbox.rejected.connect(self.cancelled)
        bbox.button(qbbox.Ok).setEnabled(False)
        layout.addWidget(bbox)

    def selection_changed(self):
        self.bbox.button(self.bbox.Ok).setEnabled(bool(self.list.selectedItems()))

    def choose_format(self):
        self.chosen_format = self.formats[self.list.selectionModel().selectedRows()[0].row()]
        self.close()

    def cancelled(self):
        self.chosen_format = None
        self.close()

def  ask_for_format(session, suffix, formats):
    dlg = AskFormatDialog(session, suffix, formats)
    dlg.exec()
    if dlg.chosen_format is None:
        from chimerax.core.errors import CancelOperation
        raise CancelOperation("User did not choose format")
    if dlg.make_default.isChecked():
        # have to force settings to notice the attribute change
        settings = session.data_formats.settings
        lookup = settings.suffix_to_format_name.copy()
        lookup[suffix] = dlg.chosen_format.name
        settings.suffix_to_format_name = lookup
    return dlg.chosen_format
