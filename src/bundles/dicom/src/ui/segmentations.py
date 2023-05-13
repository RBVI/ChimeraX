# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from Qt.QtCore import QThread, QObject, Signal, Slot, Qt
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QHeaderView
    , QWidget, QLabel, QDialog, QDialogButtonBox
    , QPushButton, QAction, QComboBox
    , QStackedWidget, QSizePolicy
)

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run
# from chimerax.core.settings import Settings

# class DICOMDatabasesSettings(Settings):
#     AUTO_SAVE = {
#         "user_accepted_tcia_tos": False
#     }

class SegmentationTool(ToolInstance):

    help = "help:user/tools/segmentations.html"
    SESSION_ENDURING = True

    def __init__(self, session = None, name = "Segmentations"):
        super().__init__(session, name)
        self._construct_ui()
        # self.settings = DICOMDatabasesSettings(self.session, "dicom databases")

    def _construct_ui(self):
        # Construct the GUI
        self.tool_window = MainToolWindow(self)
        self.parent = self.tool_window.ui_area
        self.main_layout = QVBoxLayout()

        self.view_dropdown = QComboBox(self.parent)
        self.view_dropdown.addItem("Default")
        self.view_dropdown.addItem("Orthoplanes")
        self.view_dropdown.currentIndexChanged.connect(self._on_view_changed)
        self.parent.setLayout(self.main_layout)
        self.tool_window.manage('side')

    def _on_view_changed(self):
        if self.view_dropdown.currentIndex() == 0:
            run(self.session, "dicom view default")
        else:
            run(self.session, "dicom view orthoplanes")
