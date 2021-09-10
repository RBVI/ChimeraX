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
from Qt.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QLabel

class LabelledProgressBar(QWidget):
    """Provide a labelled progress bar on all platforms, as the default QProgressBar
    does not display text, ever, on macOS."""
    def __init__(self):
        super().__init__()
        # Reach into the widget
        layout = QHBoxLayout()
        self._label = QLabel("Waiting for Results")
        self._progress_bar = QProgressBar()
        # Don't show the progress bar's text because we're going to handle it
        # with the QLabel
        self._progress_bar.setTextVisible(False)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._label)
        self.setLayout(layout)

    @property
    def text(self) -> str:
        return self._label.text()

    @text.setter
    def text(self, text: str) -> None:
        self._label.setText(text)

    @property
    def bar(self) -> QProgressBar:
        return self._progress_bar

    @property
    def value(self) -> int:
        return self._progress_bar.value()

    def setValue(self, value):
        self._progress_bar.setValue(value)

    def setMaximum(self, value):
        self._progress_bar.setMaximum(value)
