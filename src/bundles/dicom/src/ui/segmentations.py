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
    , QAbstractItemView, QSlider, QSpinBox
)

from superqt import QRangeSlider

from chimerax.core.commands import run
from chimerax.core.models import Surface, ADD_MODELS, REMOVE_MODELS
from chimerax.core.tools import ToolInstance

from chimerax.map import Volume, VolumeSurface, VolumeImage
from chimerax.map.volume import open_grids

from chimerax.ui import MainToolWindow
from chimerax.ui.open_save import SaveDialog
from chimerax.ui.widgets import ModelMenu

from ..ui.orthoplanes import Axis
from ..graphics.cylinder import SegmentationDisk
from ..dicom import modality
from ..dicom.dicom_volumes import open_dicom_grids, DICOMVolume

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
        def _not_volume_surface_or_segmentation(m):
            ok_to_list = isinstance(m, DICOMVolume)
            # This will run over all models which may not have DICOM data...
            try:
                if hasattr(m.data, "dicom_data"):
                    ok_to_list &= not m.data.dicom_data.dicom_series.modality == "SEG"
                    ok_to_list &= not m.data.reference_data
            except AttributeError:
                pass
            return ok_to_list

        self.model_menu = ModelMenu(
            self.session, self.parent, label = 'Model',
            model_types = [Volume, Surface],
            model_filter = _not_volume_surface_or_segmentation,
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
        self.edit_seg_metadata_button = QPushButton("Edit Metadata")
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
        self.slider_container = QWidget(self.parent)
        self.slider_layout = QHBoxLayout()

        self.intensity_range_label = QLabel("Intensity Range")
        self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.lower_intensity_spinbox = QSpinBox(self.slider_container)
        self.upper_intensity_spinbox = QSpinBox(self.slider_container)
        self.lower_intensity_spinbox.valueChanged.connect(self._on_spinbox_lower_intensity_range_changed)
        self.upper_intensity_spinbox.valueChanged.connect(self._on_spinbox_upper_intensity_range_changed)
        self.range_slider.sliderMoved.connect(self._on_slider_moved)
        self.slider_layout.addWidget(self.lower_intensity_spinbox)
        self.slider_layout.addWidget(self.range_slider)
        self.slider_layout.addWidget(self.upper_intensity_spinbox)
        self.slider_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_container.setLayout(self.slider_layout)
        self.main_layout.addWidget(self.intensity_range_label)
        self.main_layout.addWidget(self.slider_container)


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
        self.threshold_max = 0
        self.threshold_min = 0

        self.session.models.add(self.segmentation_cursors.values())
        self.model_added_handler = self.session.triggers.add_handler(ADD_MODELS, self._on_model_added_to_session)
        # TODO: Maybe just force the view to fourup when this tool opens?
        # TODO: if session.ui.vr_active or something, do this
        if not self.session.ui.main_window.view_layout == "orthoplanes":
        # TODO: Get this from some user preference
            run(self.session, "dicom view fourup")
        if self.session.ui.main_window.view_layout == "orthoplanes":
            self.session.ui.main_window.main_view.register_segmentation_tool(self)

        # Do the initial population of the segmentation list
        for model in self.session.models:
            if type(model) is DICOMVolume and model.is_segmentation():
                self.segmentation_list.addItem(SegmentationListItem(parent = self.segmentation_list, segmentation = model))
                if self.session.ui.main_window.view_layout == "orthoplanes":
                    self.session.ui.main_window.main_view.add_segmentation(model)
        self.segmentations_by_model = {}
        self._surface_chosen()

    def _on_spinbox_lower_intensity_range_changed(self, value):
        self.threshold_min = value
        self.range_slider.setSliderPosition([value, self.threshold_max])

    def _on_spinbox_upper_intensity_range_changed(self, value):
        self.threshold_max = value
        self.range_slider.setSliderPosition([self.threshold_min, value])

    def _on_slider_moved(self, values):
        min_, max_ = values
        self.threshold_min = int(min_)
        self.lower_intensity_spinbox.setValue(int(min_))
        self.threshold_max = int(max_)
        self.upper_intensity_spinbox.setValue(int(max_))

    def _on_model_added_to_session(self, *args):
        # If this model is a DICOM segmentation, add it to the list of segmentations
        _, model_list = args
        if model_list:
            for model in model_list:
                if type(model) is DICOMVolume and model.is_segmentation():
                    self.segmentation_list.addItem(SegmentationListItem(parent = self.segmentation_list, segmentation = model))
                    if self.session.ui.main_window.view_layout == "orthoplanes":
                        self.session.ui.main_window.main_view.add_segmentation(model)

    def delete(self):
        self.session.triggers.remove_handler(self.model_added_handler)
        if self.session.ui.main_window.view_layout == "orthoplanes":
            self.session.ui.main_window.main_view.clear_segmentation_tool()
        # When a session is closed, models are deleted before tools, so we need to
        # fail gracefully if the models have already been deleted
        try:
            self.session.models.remove(self.segmentation_cursors.values())
        except TypeError:
            pass
        super().delete()

    def _surface_chosen(self, *args):
        # If we're in the 2D view, we need to tell the orthoplane views to display
        # something new, but if we're in the 3D view, we may not need to do anything
        # except redefine the constraints of our segmentation e.g. size, spacing
        # TODO: When does this get called from the event loop / why?
        try:
            self.reference_model = self.model_menu.value
            for axis, puck in self.segmentation_cursors.items():
                puck.height = self.reference_model.data.pixel_spacing()[axis]
            # Keep the orthoplanes in sync with this menu, but don't require this menu to
            # be in sync with them
            min_ = int(self.reference_model.data.pixel_array.min())
            max_ = int(self.reference_model.data.pixel_array.max())
            self.lower_intensity_spinbox.setRange(min_, max_)
            self.upper_intensity_spinbox.setRange(min_, max_)
            self.lower_intensity_spinbox.setValue(min_)
            self.upper_intensity_spinbox.setValue(max_)
            self.range_slider.setRange(min_, max_)
            self.range_slider.setSliderPosition([min_, max_])
            self.threshold_min = min_
            self.threshold_max = max_
            self.range_slider.setTickInterval((max_ - min_) // 12)
            if self.session.ui.main_window.view_layout == "orthoplanes":
                self.session.ui.main_window.main_view.update_displayed_model(self.model_menu.value)
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

    def setMarkerRegionsToValue(self, axis, slice, markers, value=1):
        # I wasn't able to recycle code from Map Eraser here, unfortunately. Map Eraser uses
        # numpy.putmask(), which for whatever reason only wanted to work once before I had to call
        # VolumeSurface.update_surface() on the data's surface. This resulted in an expensive recalculation
        # on every loop, and it made the interface much slower -- it had to hang while the surface did many
        # redundant recalculations.
        # I can already see this becoming a massive PITA when 3D spheres get involved.
        # TODO: Many segmentations
        if not self.active_seg:
            self.addSegment()
        positions = []
        for marker in markers:
            center_x, center_y = marker.drawing_center
            radius = self.segmentation_cursors[axis].radius
            positions.append((center_x, center_y, radius))
        # TODO: Add a checkbox
        if True:
            self.active_seg.set_segment_data(axis, slice, positions, value, self.threshold_min, self.threshold_max)
        else:
            self.active_seg.set_segment_data(axis, slice, positions, value)

    def addMarkersToSegment(self, axis, slice, markers):
        self.setMarkerRegionsToValue(axis, slice, markers, 1)

    def removeMarkersFromSegment(self, axis, slice, markers):
        self.setMarkerRegionsToValue(axis, slice, markers, 0)

    def addSegment(self):
        # When the DICOMVolume creates its segmentation model, it will trigger a
        # ADD_MODEL event that we listen to above. Concerns are separated here so
        # that segmentations from files still show up in the menu.
        self.num_segmentations_created += 1
        new_seg = self.reference_model.segment(number = self.num_segmentations_created)
        num_items = self.segmentation_list.count()
        self.segmentation_list.setCurrentItem(self.segmentation_list.item(num_items - 1))
        if self.session.ui.main_window.view_layout == "orthoplanes":
            self.session.ui.main_window.main_view.add_segmentation(new_seg)

    def removeSegment(self, segments = None):
        # We don't need to listen to the REMOVE_MODEL trigger because we're going
        # to be the ones triggering it, here.
        if type(segments) is bool:
            # We got here from clicking the button...
            segments = None
        if segments is None:
            seg_item = self.segmentation_list.takeItem(self.segmentation_list.currentRow())
            segments = [seg_item.segmentation]
            seg_item.segmentation = None
            del seg_item
        if self.session.ui.main_window.view_layout == "orthoplanes":
            self.session.ui.main_window.main_view.remove_segmentation(segments[0])
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
        if self.active_seg:
            self.active_seg.active = False
        self.active_seg = segment
        self.active_seg.active = True

    def _on_active_segmentation_changed(self, new, prev):
        if new:
            self.edit_seg_metadata_button.setEnabled(True)
            self.setActiveSegment(new.segmentation)
        else:
            self.edit_seg_metadata_button.setEnabled(False)
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
            run(self.session, "dicom view default")
        if need_to_register:
            if self.session.ui.main_window.view_layout == "orthoplanes":
                # If no models are open we will not successfully change the view, so
                # we need to check the view layout before continuing!
                self.session.ui.main_window.main_view.register_segmentation_tool(self)
                if self.guidelines_checkbox.isChecked():
                    self.session.ui.main_window.main_view.toggle_guidelines()

    def set_view_dropdown(self, layout):
        if layout == "default":
            self.view_dropdown.setCurrentIndex(3)
        elif layout == "sidebyside":
            self.view_dropdown.setCurrentIndex(2)
        elif layout == "overunder":
            self.view_dropdown.setCurrentIndex(1)
        else:
            self.view_dropdown.setCurrentIndex(0)

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
        run(self.session, "help %s" % self.help)

    def _on_edit_window_ok(self) -> None:
        pass

    def editSegmentMetadata(self, _) -> None:
        pass
