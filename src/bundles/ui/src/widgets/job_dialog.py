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

from Qt.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QMenu
from Qt.QtCore import Qt

class LongJobDialog(QDialog):

    def __init__(self, *, default_duration=30, durations=[30, 60, 120, 300], job_name="job",
            question=None, title=None):
        super().__init__()
        self.duration = None

        if title is None:
            title = f"{job_name.capitalize() if job_name.islower() else job_name} still running"
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        self.setLayout(layout)
        if question is None:
            question = f"The {job_name} has been running a long time and might be hung.  Continue waiting?"
        layout.addWidget(QLabel(question))

        repeat_layout = QHBoxLayout()
        repeat_layout.addStretch(1)
        self.ask_again_box = QCheckBox("If Yes, ask again after")
        self.ask_again_box.setChecked(True)
        repeat_layout.addWidget(self.ask_again_box)
        self.duration_button = QPushButton(str(default_duration))
        menu = QMenu(self.duration_button)
        for duration in durations:
            menu.addAction(str(duration))
        menu.triggered.connect(lambda action, *args, f=self.duration_button.setText: f(action.text()))
        self.duration_button.setMenu(menu)
        repeat_layout.addWidget(self.duration_button)
        repeat_layout.addWidget(QLabel("seconds"))
        repeat_layout.addStretch(1)
        layout.addLayout(repeat_layout)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Yes | qbbox.No)
        bbox.accepted.connect(self.set_duration)
        bbox.accepted.connect(self.close)
        bbox.rejected.connect(self.close)
        layout.addWidget(bbox)

    def set_duration(self):
        if self.ask_again_box.isChecked():
            self.duration = float(self.duration_button.text())
        else:
            self.duration = True

    def run(self):
        self.exec()
        return self.duration

