# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from Qt.QtCore import Qt
from Qt.QtWidgets import QWidget, QFormLayout, QLabel

class BlastProteinFormWidget(QWidget):
    def __init__(self, label, input_widget, parent):
        super().__init__()
        layout = QFormLayout()
        self._label = QLabel(label)
        self._input_widget = input_widget(parent)
        layout.setWidget(0, QFormLayout.LabelRole, self._label)
        layout.setWidget(0, QFormLayout.FieldRole, self._input_widget)
        layout.setLabelAlignment(Qt.AlignLeft)
        layout.setFormAlignment(Qt.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

    @property
    def input_widget(self) -> QWidget:
        return self._input_widget

    @property
    def label(self) -> QLabel:
        return self._label
