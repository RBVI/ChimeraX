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
    , QListWidget, QListWidgetItem, QFileDialog
    , QAbstractItemView
)

from chimerax.ui.widgets import ModelMenu
from chimerax.map import Volume, VolumeSurface, VolumeImage
from chimerax.map.volume import open_grids
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.ui.open_save import SaveDialog
from chimerax.core.commands import run
from chimerax.core.models import Surface
from ..ui.orthoplanes import Axis
from ..graphics.cylinder import SegmentationDisk
from ..dicom.dicom_models import DicomSegmentation

class SegmentationListItem(QListWidgetItem):
    def __init__(self, parent, segmentation):
        super().__init__(parent)
        self.segmentation = segmentation
        self.setText(self.segmentation.name)

class SegmentationTool(ToolInstance):
    # TODO: Sphere cursor for 2D, extend to VR

    help = "help:user/tools/segmentations.html"
    SESSION_ENDURING = True
    num_segmentations_created = 0

    def __init__(self, session = None, name = "Segmentations"):
        super().__init__(session, name)
        self._construct_ui()

    def _construct_ui(self):
        # Construct the GUI
        self.tool_window = MainToolWindow(self)
        self.parent = self.tool_window.ui_area
        self.main_layout = QVBoxLayout()
        self.active_seg: Volume | None = None

        self.view_dropdown_container = QWidget(self.parent)
        self.view_dropdown_layout = QHBoxLayout()
        self.view_dropdown_label = QLabel("View Layout")
        self.view_dropdown = QComboBox(self.parent)
        self.view_dropdown.addItem("4 x 4 (Desktop)")
        self.view_dropdown.addItem("3D Over Orthoplanes (Desktop)")
        self.view_dropdown.addItem("3D Beside Orthoplanes (Desktop)")
        self.view_dropdown.addItem("3D Only (VR)")
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

        self.guidelines_checkbox = QCheckBox("Plane Guidelines")
        self.control_checkbox_layout.addWidget(self.model_menu.frame)
        self.control_checkbox_layout.addWidget(self.guidelines_checkbox)
        self.guidelines_checkbox.stateChanged.connect(self._on_show_guidelines_checkbox_changed)
        self.control_checkbox_container.setLayout(self.control_checkbox_layout)

        self.control_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.control_checkbox_layout.setSpacing(0)

        self.add_remove_save_container = QWidget()
        self.add_remove_save_layout = QHBoxLayout()
        self.add_seg_button = QPushButton("Add")
        self.remove_seg_button = QPushButton("Remove")
        self.edit_seg_metadata_button = QPushButton("Edit")
        self.save_seg_button = QPushButton("Save")
        self.help_button = QPushButton("Help")

        self.add_remove_save_layout.addWidget(self.add_seg_button)
        self.add_remove_save_layout.addWidget(self.remove_seg_button)
        self.add_remove_save_layout.addWidget(self.edit_seg_metadata_button)
        self.add_remove_save_layout.addWidget(self.save_seg_button)
        self.add_remove_save_layout.addStretch()
        self.add_remove_save_layout.addWidget(self.help_button)
        self.add_remove_save_container.setLayout(self.add_remove_save_layout)
        self.add_seg_button.clicked.connect(self.addSegment)
        self.remove_seg_button.clicked.connect(self.removeSegment)
        self.save_seg_button.clicked.connect(self.saveSegment)
        self.edit_seg_metadata_button.clicked.connect(self.editSegmentMetadata)
        self.help_button.clicked.connect(self.showHelp)

        self.add_remove_save_layout.setContentsMargins(0, 0, 0, 0)
        self.add_remove_save_layout.setSpacing(0)

        self.main_layout.addWidget(self.view_dropdown_container)
        self.main_layout.addWidget(self.control_checkbox_container)

        self.segmentation_list_label = QLabel("Segmentations")
        self.segmentation_list = QListWidget(parent = self.parent)
        self.segmentation_list.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.MinimumExpanding)
        self.segmentation_list.currentItemChanged.connect(self._on_active_segmentation_changed)
        self.main_layout.addWidget(self.segmentation_list_label)
        self.main_layout.addWidget(self.add_remove_save_container)
        self.main_layout.addWidget(self.segmentation_list)

        self.main_layout.addStretch()
        self.main_layout.setContentsMargins(6, 0, 6, 6)
        self.main_layout.setSpacing(0)

        self.parent.setLayout(self.main_layout)
        self.tool_window.manage('side')
        self.segmentation_cursors = {}
        self._create_2d_segmentation_pucks()
        self.segmentations = {}
        self.current_segmentation = None
        self.reference_model = None

        self.session.models.add(self.segmentation_cursors.values())
        # TODO: Maybe just force the view to fourup when this tool opens?
        # TODO: if session.ui.vr_active or something, do this
        if not self.session.ui.main_window.view_layout == "orthoplanes":
        # TODO: Get this from some user preference
            run(self.session, "dicom view fourup")
        if self.session.ui.main_window.view_layout == "orthoplanes":
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
                    for axis, puck in self.segmentation_cursors.items():
                        puck.height = new_model.data.pixel_spacing()[axis]
                        # Set by orthoplanes.py
                        #puck.origin = [x for x in medical_image_data.origin()]
            self.reference_model = new_drawing
        except AttributeError: # No more volumes!
            pass

    def _create_2d_segmentation_pucks(self, initial_display = False) -> None:
        self.segmentation_cursors = {
            Axis.AXIAL:    SegmentationDisk(self.session, Axis.AXIAL, height=5)
            , Axis.CORONAL:  SegmentationDisk(self.session, Axis.CORONAL, height=5)
            , Axis.SAGITTAL: SegmentationDisk(self.session, Axis.SAGITTAL, height=5)
        }
        for cursor in self.segmentation_cursors.values():
            cursor.display = initial_display

    def _set_data_in_puck(self, grid, axis, slice, left_offset: int, bottom_offset: int, radius: int, value: int) -> None:
        # TODO: Preserve the happiest path. If the radius of the segmentation overlay is
        #  less than the radius of one voxel, there's no need to go through all the rigamarole.
        #  grid.data.segment_array[slice][left_offset][bottom_offset] = 1
        x_max, y_max, z_max = grid.data.size
        x_step, y_step, z_step = grid.data.step
        if axis == Axis.AXIAL:
            slice = grid.data.segment_array[slice]
            vertical_max = y_max - 1
            vertical_step = y_step
            horizontal_max = x_max - 1
            horizontal_step = x_step
        elif axis == Axis.CORONAL:
            slice = grid.data.segment_array[:, slice, :]
            vertical_max = z_max - 1
            vertical_step = z_step
            horizontal_max = x_max - 1
            horizontal_step = x_step
        else:
            slice = grid.data.segment_array[:, :, slice]
            vertical_max = z_max - 1
            vertical_step = z_step
            horizontal_max = y_max - 1
            horizontal_step = y_step
        scaled_radius = round(radius / horizontal_step)
        x = 0
        y = round(radius)
        d = 1 - y
        while y > x:
            if d < 0:
                d += 2 * x + 3
            else:
                d += 2 * (x - y) + 5
                y -= 1
            x += 1
            scaled_horiz_x = round(x / horizontal_step)
            scaled_vert_x = round(x / vertical_step)
            scaled_horiz_y = round(y / horizontal_step)
            scaled_vert_y = round(y / vertical_step)
            x_start = max(left_offset - scaled_horiz_x, 0)
            x_end = min(left_offset + scaled_horiz_x, horizontal_max)
            y_start = max(bottom_offset - scaled_vert_y, 0)
            y_end = min(bottom_offset + scaled_vert_y, vertical_max)
            slice[y_start][x_start:x_end] = value
            slice[y_end][x_start:x_end] = value
            # Try to account for the fact that with spacings < 1 some lines get skipped, even if it
            # causes redundant writes
            slice[y_start + 1][x_start:x_end] = value
            slice[y_end - 1][x_start:x_end] = value
            x_start = max(left_offset - scaled_horiz_y, 0)
            x_end = min(left_offset + scaled_horiz_y, horizontal_max)
            y_start = max(bottom_offset - scaled_vert_x, 0)
            y_end = min(bottom_offset + scaled_vert_x, vertical_max)
            slice[y_start][x_start:x_end] = value
            slice[y_end][x_start:x_end] = value
            # Try to account for the fact that with spacings < 1 some lines get skipped, even if it
            # causes redundant writes
            slice[y_start + 1][x_start:x_end] = value
            slice[y_end - 1][x_start:x_end] = value
        slice[bottom_offset][left_offset - scaled_radius:left_offset + scaled_radius] = value

    def make_puck_visible(self, axis):
        if axis in self.segmentation_cursors:
            self.segmentation_cursors[axis].display = True

    def make_puck_invisible(self, axis):
        if axis in self.segmentation_cursors:
            self.segmentation_cursors[axis].display = False

    def setGuidelineCheckboxValue(self, visible):
        if visible:
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.Unchecked
        self.guidelines_checkbox.setCheckState(state)

    def addMarkersToSegment(self, axis, slice, positions):
        # I wasn't able to recycle code from Map Eraser here, unfortunately. Map Eraser uses
        # numpy.putmask(), which for whatever reason only wanted to work once before I had to call
        # VolumeSurface.update_surface() on the data's surface. This resulted in an expensive recalculation
        # on every loop, and it made the interface much slower -- it had to hang while the surface did many
        # redundant recalculations.
        # I can already see this becoming a massive PITA when 3D spheres get involved.
        # TODO: Many segmentations
        if not self.active_seg:
            self.session.logger.error("No active segmentation!")
            return
        for position in positions:
            center_x, center_y = position.drawing_center
            radius = self.segmentation_cursors[axis].radius
            self._set_data_in_puck(self.active_seg, axis, slice, round(center_x), round(center_y), radius, 1)
        self.active_seg.data.values_changed()

    def removeMarkersFromSegment(self, axis, slice, positions):
        if not self.active_seg:
            self.session.logger.error("No active segmentation!")
            return
        for position in positions:
            center_x, center_y = position.drawing_center
            radius = self.segmentation_cursors[axis].radius
            self._set_data_in_puck(self.active_seg, axis, slice, round(center_x), round(center_y), radius, 0)
        self.active_seg.data.values_changed()
        # self.active_seg.update_drawings()

    def addSegment(self):
        # TODO: Create an empty DICOM Volume and add it to both
        # the reference_model's child drawings and the
        new_seg = self.reference_model.parent.data.segment(number = self.num_segmentations_created + 1)
        self.num_segmentations_created += 1
        new_seg_model = open_grids(self.session, [new_seg], name = "new segmentation")[0]
        self.session.models.add(new_seg_model)
        new_seg_model[0].set_parameters(surface_levels=[0.501])
        self.segmentation_list.addItem(SegmentationListItem(parent = self.segmentation_list, segmentation = new_seg_model[0]))
        num_items = self.segmentation_list.count()
        self.segmentation_list.setCurrentItem(self.segmentation_list.item(num_items - 1))

    def removeSegment(self, segments = None):
        if type(segments) is bool:
            # We got here from clicking the button...
            segments = None
        if segments is None:
            seg_item = self.segmentation_list.takeItem(self.segmentation_list.currentRow())
            segments = [seg_item.segmentation]
            seg_item.segmentation = None
            del seg_item
        self.session.models.remove(segments)

    def saveSegment(self, segments = None):
        if type(segments) is bool:
            # Why in world would this be a bool?? Go home Qt, you're drunk.
            segments = None
        if segments is None:
            segments = self.segmentation_list.selectedItems()
        sd = SaveDialog(self.session, parent=self.tool_window.ui_area)
        sd.setNameFilter("DICOM (*.dcm)")
        sd.setDefaultSuffix("dcm")
        if not sd.exec():
            return
        filename = sd.selectedFiles()[0]
        self.active_seg.data.save(filename)

    def setActiveSegment(self, segment):
        self.active_seg = segment

    def _on_active_segmentation_changed(self, new, prev):
        if new:
            self.setActiveSegment(new.segmentation)
        else:
            self.setActiveSegment(None)

    def _on_view_changed(self):
        need_to_register = False
        if self.view_dropdown.currentIndex() == 0:
            run(self.session, "dicom view fourup")
            need_to_register = True
        elif self.view_dropdown.currentIndex() == 1:
            run(self.session, "dicom view overunder")
            need_to_register = True
        elif self.view_dropdown.currentIndex() == 2:
            run(self.session, "dicom view sidebyside")
            need_to_register = True
        else:
            run(self.session, "dicom view fourup")
        if need_to_register:
            self.session.ui.main_window.main_view.register_segmentation_tool(self)
            if self.guidelines_checkbox.isChecked():
                self.session.ui.main_window.main_view.toggle_guidelines()

    def setPuckHeight(self, axis, height):
        self.segmentation_cursors[axis].height = height

    def _on_show_guidelines_checkbox_changed(self):
        if self.session.ui.main_window.view_layout == "orthoplanes":
            self.session.ui.main_window.main_view.register_segmentation_tool(self)
            self.session.ui.main_window.main_view.toggle_guidelines()
        # else if it's just the main window, save that the user wanted it to be displayed
        # and on change enable it?

    def setCursorOffsetFromOrigin(self, axis, offset):
        offsets = self.segmentation_cursors[axis].origin
        offsets[axis] = offset
        self.segmentation_cursors[axis].origin = offsets

    def showHelp(self, _) -> None:
        run(self.session, "help segmentations")

    def _on_edit_window_ok(self) -> None:
        pass

    def editSegmentMetadata(self, _) -> None:
        pass
