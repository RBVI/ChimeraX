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
    def __init__(self, label, input_widget = None, parent = None):
        super().__init__()
        self.layout = QFormLayout()
        self._label = QLabel(label)
        if input_widget:
            self._input_widget = input_widget(parent)
        self.layout.setWidget(0, QFormLayout.LabelRole, self._label)
        if input_widget:
            self.layout.setWidget(0, QFormLayout.FieldRole, self._input_widget)
        self.layout.setLabelAlignment(Qt.AlignLeft)
        self.layout.setFormAlignment(Qt.AlignLeft)
        # If the second value is not at least 4 then ChainMenuButton stops being
        # rendered in native macOS style.
        self.layout.setContentsMargins(0,4,0,0)
        self.layout.setSpacing(2)
        self.setLayout(self.layout)

    @property
    def input_widget(self) -> QWidget:
        return self._input_widget

    @property
    def label(self) -> QLabel:
        return self._label

    @input_widget.setter
    def input_widget(self, widget) -> None:
        self._input_widget = widget
        self.layout.setWidget(0, QFormLayout.FieldRole, widget)
