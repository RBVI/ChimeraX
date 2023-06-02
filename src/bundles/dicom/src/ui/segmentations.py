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
    , QStackedWidget, QSizePolicy, QCheckBox
)

from chimerax.ui.widgets import ModelMenu
from chimerax.map import Volume, VolumeSurface, VolumeImage
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run
from chimerax.core.models import Surface
from .view import dicom_view
from ..ui.orthoplanes import Axis
from ..graphics.cylinder import SegmentationDisk

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
        dicom_view(self.session, "orthoplanes")
        self.view_dropdown.addItem("Orthoplanes")
        if self.session.ui.main_window.view_layout == "fourup":
            self.view_dropdown.setCurrentIndex(1)
        self.view_dropdown.currentIndexChanged.connect(self._on_view_changed)

        def _not_volume_surface(m):
            return not isinstance(m, VolumeSurface)

        self.model_menu = ModelMenu(
            self.session, self.parent, label = 'Model',
            model_types = [Volume, Surface],
            model_filter = _not_volume_surface,
            model_chosen_cb = self._surface_chosen
        )

        self.control_checkbox_container = QWidget()
        self.control_checkbox_layout = QHBoxLayout()

        self.guidelines_checkbox = QCheckBox("Toggle Plane Guidelines")
        self.control_checkbox_layout.addWidget(self.model_menu.frame)
        self.control_checkbox_layout.addWidget(self.guidelines_checkbox)
        self.guidelines_checkbox.stateChanged.connect(self._on_show_guidelines_checkbox_changed)
        self.control_checkbox_container.setLayout(self.control_checkbox_layout)

        self.main_layout.addWidget(self.view_dropdown)
        self.main_layout.addWidget(self.control_checkbox_container)
        self.main_layout.addStretch()
        self.parent.setLayout(self.main_layout)
        self.tool_window.manage('side')
        self.segmentation_cursors = {
            Axis.AXIAL:    SegmentationDisk(self.session, Axis.AXIAL, height=5)
            , Axis.CORONAL:  SegmentationDisk(self.session, Axis.CORONAL, height=5)
            , Axis.SAGGITAL: SegmentationDisk(self.session, Axis.SAGGITAL, height=5)
        }

        self.session.models.add(self.segmentation_cursors.values())
        # TODO: Maybe just force the view to fourup when this tool opens?
        if self.session.ui.main_window.view_layout == "fourup":
            self.session.ui.main_window.main_view.register_segmentation_tool(self)

    def _surface_chosen(self, *args):
        # TODO: When does this get called from the event loop / why?
        try:
            new_model = self.model_menu.value
            if new_model is not None:
                new_drawing = None
                for d in self.model_menu.value._child_drawings:
                    if type(d) is VolumeImage:
                        new_drawing = d
                    medical_image_data = new_model.data.dicom_data
                    ortho_pos = new_drawing._rendering_options.orthoplane_positions
                    for axis, puck in self.segmentation_cursors.items():
                        puck.height = medical_image_data.pixel_spacing()[axis]
                        # Set by orthoplanes.py
                        #puck.origin = [x for x in medical_image_data.origin()]

        except AttributeError: # No more volumes!
            pass

    def _on_view_changed(self):
        if self.view_dropdown.currentIndex() == 0:
            run(self.session, "dicom view default")
        else:
            run(self.session, "dicom view orthoplanes")

    def _on_show_guidelines_checkbox_changed(self):
        if self.session.ui.main_window.view_layout == "fourup":
            self.session.ui.main_window.main_view.register_segmentation_tool(self)
            self.session.ui.main_window.main_view.toggle_guidelines()
        # else if it's just the main window, save that the user wanted it to be displayed
        # and on change enable it?

    def setCursorOffsetFromOrigin(self, axis, offset):
        offsets = [0, 0, 0]
        offsets[axis] = offset
        self.segmentation_cursors[axis].origin = offsets
