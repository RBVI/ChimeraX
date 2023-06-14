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
from chimerax.map.volume import open_grids
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run
from chimerax.core.models import Surface
from .view import dicom_view
from ..ui.orthoplanes import Axis
from ..graphics.cylinder import SegmentationDisk
from ..dicom.dicom_models import DicomSegmentation

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

        self.view_dropdown_container = QWidget(self.parent)
        self.view_dropdown_layout = QHBoxLayout()
        self.view_dropdown_label = QLabel("View Layout")
        self.view_dropdown = QComboBox(self.parent)
        self.view_dropdown.addItem("Default (VR)")
        dicom_view(self.session, "orthoplanes")
        self.view_dropdown.addItem("Orthoplanes (Desktop)")
        if self.session.ui.main_window.view_layout == "fourup":
            self.view_dropdown.setCurrentIndex(1)
        self.view_dropdown.currentIndexChanged.connect(self._on_view_changed)
        self.view_dropdown_layout.addWidget(self.view_dropdown_label)
        self.view_dropdown_layout.addWidget(self.view_dropdown, 1)
        self.view_dropdown_container.setLayout(self.view_dropdown_layout)

        self.view_dropdown_layout.setContentsMargins(0, 0, 0, 0)
        self.view_dropdown_layout.setSpacing(0)
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

        self.control_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.control_checkbox_layout.setSpacing(0)

        self.add_remove_container = QWidget()
        self.add_remove_layout = QHBoxLayout()
        self.add_seg_button = QPushButton("Add Segmentation")
        self.remove_seg_button = QPushButton("Remove Segmentation")

        self.add_remove_layout.addWidget(self.add_seg_button)
        self.add_remove_layout.addWidget(self.remove_seg_button)
        self.add_remove_container.setLayout(self.add_remove_layout)
        self.add_seg_button.clicked.connect(self.addSegment)
        self.remove_seg_button.clicked.connect(self.removeSegment)

        self.add_remove_layout.setContentsMargins(0, 0, 0, 0)
        self.add_remove_layout.setSpacing(0)

        self.main_layout.addWidget(self.view_dropdown_container)
        self.main_layout.addWidget(self.control_checkbox_container)
        self.main_layout.addWidget(self.add_remove_container)

        self.main_layout.addStretch()
        self.main_layout.setContentsMargins(6, 0, 6, 0)
        self.main_layout.setSpacing(0)

        self.parent.setLayout(self.main_layout)
        self.tool_window.manage('side')
        self.segmentation_cursors = {
            Axis.AXIAL:    SegmentationDisk(self.session, Axis.AXIAL, height=5)
            , Axis.CORONAL:  SegmentationDisk(self.session, Axis.CORONAL, height=5)
            , Axis.SAGGITAL: SegmentationDisk(self.session, Axis.SAGGITAL, height=5)
        }
        self.segmentations = {}
        self.current_segmentation = None
        self.reference_model = None

        self.session.models.add(self.segmentation_cursors.values())
        # TODO: Maybe just force the view to fourup when this tool opens?
        if self.session.ui.main_window.view_layout == "fourup":
            self.session.ui.main_window.main_view.register_segmentation_tool(self)
        self._surface_chosen()

    def delete(self):
        self.session.ui.main_window.main_view.clear_segmentation_tool()
        self.session.models.remove(self.segmentation_cursors.values())
        super().delete()

    def _surface_chosen(self, *args):
        # If we're in the 2D view, we need to tell the orthoplane views to display
        # something new, but if we're in the 3D view, we may not need to do anything
        # except redefine the constraints of our segmentation e.g. size, spacing
        # TODO: When does this get called from the event loop / why?
        try:
            new_model = self.model_menu.value
            if new_model is not None:
                new_drawing = None
                for d in self.model_menu.value._child_drawings:
                    if type(d) is VolumeImage:
                        new_drawing = d
                    medical_image_data = new_model.data.dicom_data
                    for axis, puck in self.segmentation_cursors.items():
                        puck.height = medical_image_data.pixel_spacing()[axis]
                        # Set by orthoplanes.py
                        #puck.origin = [x for x in medical_image_data.origin()]
            self.reference_model = new_drawing
        except AttributeError: # No more volumes!
            pass

    def _set_data_in_puck(self, grid, axis, slice, left_offset: int, bottom_offset: int, radius: int) -> None:
        # TODO: Preserve the happiest path. If the radius of the segmentation overlay is
        #  less than the radius of one voxel, there's no need to go through all the rigamarole.
        #  grid.data.segment_array[slice][left_offset][bottom_offset] = 1
        x_max, y_max, z_max = grid.data.size
        if axis == Axis.AXIAL:
            slice = grid.data.segment_array[slice]
            vertical_max = y_max - 1
            horizontal_max = x_max - 1
        elif axis == Axis.CORONAL:
            slice = grid.data.segment_array[:, slice, :]
            vertical_max = z_max - 1
            horizontal_max = x_max - 1
        else:
            slice = grid.data.segment_array[:, :, slice]
            vertical_max = z_max - 1
            horizontal_max = y_max - 1
        x = 0
        y = radius
        d = 1 - y
        while y > x:
            if d < 0:
                d += 2 * x + 3
            else:
                d += 2 * (x - y) + 5
                y -= 1
            x += 1
            x_start = max(left_offset - x, 0)
            x_end = min(left_offset + x, horizontal_max)
            y_start = max(bottom_offset - round(y), 0)
            y_end = min(bottom_offset + round(y), vertical_max)
            slice[y_start][x_start:x_end] = 1
            slice[y_end][x_start:x_end] = 1
            x_start = max(left_offset - round(y), 0)
            x_end = min(left_offset + round(y), horizontal_max)
            y_start = max(bottom_offset - x, 0)
            y_end = min(bottom_offset + x, vertical_max)
            slice[y_start][x_start:x_end] = 1
            slice[y_end][x_start:x_end] = 1

    def addMarkersToSegment(self, axis, slice, positions):
        # I wasn't able to recycle code from Map Eraser here, unfortunately. Map Eraser uses
        # numpy.putmask(), which for whatever reason only wanted to work once before I had to call
        # VolumeSurface.update_surface() on the data's surface. This resulted in an expensive recalculation
        # on every loop, and it made the interface much slower -- it had to hang while the surface did many
        # redundant recalculations.
        # I can already see this becoming a massive PITA when 3D spheres get involved.
        for position in positions:
            center_x, center_y = position.drawing_center
            radius = self.segmentation_cursors[axis].radius
            self._set_data_in_puck(self.active_seg, axis, slice, round(center_x), round(center_y), radius)
        self.active_seg.data.values_changed()

    def removeMarkersFromSegment(self, axis, slice, positions):
        pass

    def addSegment(self):
        # TODO: Create an empty DICOM Volume and add it to both
        # the reference_model's child drawings and the
        new_seg = DicomSegmentation(self.reference_model.parent.data.dicom_data)
        new_seg_model = open_grids(self.session, [new_seg], name = "new segmentation")[0]
        self.active_seg = new_seg_model[0]
        self.session.models.add(new_seg_model)
        # TODO: This forces the orthoplanes to be visible for some reason??
        #self.session.ui.main_window.main_view.add_segmentation(new_seg_model)

    def removeSegment(self, segments = None):
        if segments is None:
            # TODO: Get the currently highlighted segment from the table
            pass
        if segments:
            self.session.models.remove(segments)

    def _on_view_changed(self):
        if self.view_dropdown.currentIndex() == 0:
            run(self.session, "dicom view default")
        else:
            run(self.session, "dicom view orthoplanes")
            self.session.ui.main_window.main_view.register_segmentation_tool(self)
            if self.guidelines_checkbox.isChecked():
                self.session.ui.main_window.main_view.toggle_guidelines()

    def setPuckHeight(self, axis, height):
        self.segmentation_cursors[axis].height = height

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
