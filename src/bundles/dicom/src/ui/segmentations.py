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
import os
import sys

from enum import IntEnum

from Qt.QtCore import QThread, QObject, Signal, Slot, Qt
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QHeaderView
    , QWidget, QLabel, QDialog, QDialogButtonBox
    , QPushButton, QAction, QComboBox
    , QStackedWidget, QSizePolicy, QCheckBox
    , QListWidget, QListWidgetItem, QFileDialog
    , QAbstractItemView, QSlider, QSpinBox
    , QStyle, QTabWidget, QToolButton
)
from Qt.QtGui import QImage, QPixmap

from superqt import QRangeSlider

from chimerax.core.commands import run
from chimerax.core.models import Surface, ADD_MODELS, REMOVE_MODELS
from chimerax.core.tools import ToolInstance
from chimerax.core.settings import Settings

from chimerax.geometry import Place, translation

from chimerax.map import Volume, VolumeSurface, VolumeImage
from chimerax.map.volume import open_grids

from chimerax.ui import MainToolWindow
from chimerax.ui.font import shrink_font
from chimerax.ui.open_save import SaveDialog
from chimerax.ui.options import SettingsPanel, BooleanOption, SymbolicEnumOption, IntOption
from chimerax.ui.widgets import ModelMenu
from chimerax.ui.icons import get_qt_icon

from chimerax.vive.vr import vr_camera as steamvr_camera
from chimerax.vive.vr import vr_button as steamvr_button
from chimerax.vive.vr import SteamVRCamera
from chimerax.vive.xr import vr_camera as openxr_camera
from chimerax.vive.xr import vr_button as openxr_button
from chimerax.vive.xr import OpenXRCamera

from ..ui.orthoplanes import Axis
from ..graphics.cylinder import SegmentationDisk
from ..graphics.sphere import SegmentationSphere
from ..dicom import modality
from ..dicom.dicom_volumes import open_dicom_grids, DICOMVolume

class SegmentationListItem(QListWidgetItem):
    def __init__(self, parent, segmentation):
        super().__init__(parent)
        self.segmentation = segmentation
        self.setText(self.segmentation.name)

# Use these enums to populate the dropdowns so that the order is consistent
class ViewMode(IntEnum):
    TWO_BY_TWO = 0
    ORTHOPLANES_OVER_3D = 1
    ORTHOPLANES_BESIDE_3D = 2
    DEFAULT_DESKTOP = 3
    DEFAULT_VR = 4

    def __str__(self):
        if self.name == "TWO_BY_TWO":
            return "2 x 2 (desktop)"
        elif self.name == "ORTHOPLANES_OVER_3D":
            return "3D over slices (desktop)"
        elif self.name == "ORTHOPLANES_BESIDE_3D":
            return "3D beside slices (desktop)"
        elif self.name == "DEFAULT_DESKTOP":
            return "3D only (desktop)"
        elif self.name == "DEFAULT_VR":
            return "3D only (VR)"
        return "%s: Set a value to return for the name of this EnumItem" % self.name

class ImageFormat(IntEnum):
    DICOM = 0
    NIFTI = 1
    NRRD = 2

    def __str__(self):
        if self.name == "NIFTI":
            return "NIfTI"
        return self.name

class MouseAction(IntEnum):
    NONE = 0
    ADD_TO_SEGMENTATION = 1
    MOVE_SPHERE = 2
    RESIZE_SPHERE = 3
    ERASE_FROM_SEGMENTATION = 4

    def __str__(self):
        return " ".join(self.name.split('_')).lower()

class HandAction(IntEnum):
    NONE = 0
    RESIZE_CURSOR = 1
    MOVE_CURSOR = 2
    ADD_TO_SEGMENTATION = 3
    ERASE_FROM_SEGMENTATION = 4

    def __str__(self):
        return " ".join(self.name.split('_')).lower()

class Handedness(IntEnum):
    LEFT = 0
    RIGHT = 1

    def __str__(self):
        return self.name.title()

class _SegmentationToolSettings(Settings):
    EXPLICIT_SAVE = {
        'start_vr_automatically': False
        , 'set_mouse_modes_automatically': False
        , 'set_hand_modes_automatically': False
        , 'default_view': 0 # 4 x 4
        , 'default_file_format': 0 # DICOM
        , 'default_segmentation_opacity': 80 # %
        , 'mouse_3d_right_click': MouseAction.ADD_TO_SEGMENTATION
        , 'mouse_3d_middle_click': MouseAction.MOVE_SPHERE
        , 'mouse_3d_scroll': MouseAction.RESIZE_SPHERE
        , 'mouse_3d_left_click': MouseAction.NONE
        , 'vr_thumbstick': HandAction.RESIZE_CURSOR
        , 'vr_trigger': HandAction.ADD_TO_SEGMENTATION
        , 'vr_grip': HandAction.MOVE_CURSOR
        , 'vr_a_button': HandAction.ERASE_FROM_SEGMENTATION
        , 'vr_b_button': HandAction.NONE
        , 'vr_handedness': Handedness.RIGHT
    }

_seg_tool_settings = None

def get_settings(session):
    global _seg_tool_settings
    if _seg_tool_settings is None:
        _seg_tool_settings = _SegmentationToolSettings(session, "Segmentation Tool")
    return _seg_tool_settings
        

class SegmentationToolControlsDialog(QDialog):
    right_hand_image = QImage(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icons", "right_controller.png"))
    # left_hand_image = QImage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "right_controller.png"))
    mouse_image = QImage(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icons", "mouse.png"))

    def __init__(self, parent, session):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Segmentation Tool Settings")
        self.setLayout(QVBoxLayout())
        self.tab_widget = QTabWidget(self)
        self.layout().addWidget(self.tab_widget)
        self._add_settings_tab()
        self._add_mouse_2d_tab()
        self._add_mouse_3d_tab()
        self._add_vr_tab()

    def _add_settings_tab(self):
        settings = get_settings(self.session)
        self.settings_container = QWidget(self)
        self.settings_container.setLayout(QVBoxLayout())
        self.panel = SettingsPanel()
        self.panel.add_option(
            SymbolicEnumOption(
                name = "Default layout"
                , default = None
                , attr_name = "default_view"
                , settings = settings
                , callback = None
                , labels = [str(mode) for mode in ViewMode]
                , values = [mode.value for mode in ViewMode]
            )
        )
        self.panel.add_option(
            BooleanOption(
                name = "Set 3D mouse modes when the desktop 3D-only layout is chosen"
                , default = None
                , attr_name = "set_mouse_modes_automatically"
                , settings = settings
                , callback = None
            )
        )
        self.panel.add_option(
            BooleanOption(
                name = "Start VR when the VR layout is chosen"
                , default = None
                , attr_name = "start_vr_automatically"
                , settings = settings
                , callback = None
            )
        )
        self.panel.add_option(
            BooleanOption(
                name = "Set VR controller modes when the VR layout is chosen"
                , attr_name = "set_hand_modes_automatically"
                , settings = settings
                , callback = None
                , default = None
            )
        )
        self.panel.add_option(
            IntOption(
                "Segmentation opacity"
                , default = None
                , settings = settings
                , min = 0
                , max = 100
                , callback = None
                , attr_name = "default_segmentation_opacity"
            )
        )
        self.panel.add_option(
            SymbolicEnumOption(
                name = "Format for saving segmentations"
                , default = None
                , settings = settings
                , callback = None
                , labels= [str(format) for format in ImageFormat]
                , values = [format.value for format in ImageFormat]
                , attr_name = "default_file_format"
            )
        )
        self.settings_container.layout().addWidget(self.panel)
        self.tab_widget.addTab(self.settings_container, "General")

    def _add_mouse_2d_tab(self):
        self.mouse_2d_outer_widget = QWidget(self)
        self.mouse_2d_outer_layout = QHBoxLayout()
        self.mouse_2d_outer_widget.setLayout(self.mouse_2d_outer_layout)

        self.mouse_2d_image_widget = QLabel(self)
        self.mouse_2d_image_widget.setPixmap(QPixmap.fromImage(self.mouse_image.scaledToWidth(350, Qt.TransformationMode.SmoothTransformation)))

        self.mouse_2d_image_container = QWidget()
        self.mouse_2d_image_container_layout = QVBoxLayout()
        self.mouse_2d_image_container_layout.setSpacing(0)
        self.mouse_2d_image_container_layout.setContentsMargins(0, 0, 0, 0)
        self.mouse_2d_image_container.setLayout(self.mouse_2d_image_container_layout)
        self.mouse_2d_image_container_layout.addWidget(self.mouse_2d_image_widget)
        self.mouse_2d_image_container_layout.addStretch(1)
        self.mouse_2d_outer_layout.addWidget(self.mouse_2d_image_container)

        self.mouse_control_2d_dropdown_container = QWidget()
        self.mouse_control_2d_dropdown_container_layout = QVBoxLayout()
        self.mouse_control_2d_dropdown_container.setLayout(self.mouse_control_2d_dropdown_container_layout)
        windows_spacings = [50, 12, 12, 12]
        mac_spacings = [50, 12, 12, 12]
        linux_spacings = [52, 14, 18, 14]
        if sys.platform == "win32":
            spacings = windows_spacings
        elif sys.platform == "darwin":
            spacings = mac_spacings
        else:
            spacings = linux_spacings
        control_labels = [
            QLabel("Zoom slice")
            , QLabel("Pan slice")
            , QLabel("Zoom slice\n(+Shift) Resize segmentation circle")
            , QLabel("Add to segmentation\n(+Shift) Erase from segmentation")
        ]
        for i in range(4):
            self.mouse_control_2d_dropdown_container_layout.addSpacing(spacings[i])
            self.mouse_control_2d_dropdown_container_layout.addWidget(control_labels[i])
        self.mouse_control_2d_dropdown_container_layout.addStretch()
        self.mouse_control_2d_dropdown_container_layout.setContentsMargins(6, 0, 0, 0)
        self.mouse_control_2d_dropdown_container_layout.setSpacing(0)
        self.mouse_2d_outer_layout.addWidget(self.mouse_control_2d_dropdown_container)
        self.mouse_2d_outer_layout.addStretch(1)
        self.tab_widget.addTab(self.mouse_2d_outer_widget, "Mouse (2D Slices)")

    def _add_mouse_3d_tab(self):
        self.mouse_3d_widget = QWidget(self)
        self.mouse_3d_layout = QVBoxLayout()
        self.mouse_3d_layout.setSpacing(0)
        self.mouse_3d_widget.setLayout(self.mouse_3d_layout)

        self.mouse_3d_image_controls_container = QWidget(self)
        self.mouse_3d_image_controls_container_layout = QHBoxLayout()
        self.mouse_3d_image_controls_container_layout.setContentsMargins(0, 0, 0, 0)
        self.mouse_3d_image_controls_container_layout.setSpacing(0)
        self.mouse_3d_image_controls_container.setLayout(self.mouse_3d_image_controls_container_layout)

        self.mouse_3d_layout.addWidget(self.mouse_3d_image_controls_container)

        self.mouse_controls_3d_label_list_container = QWidget()
        self.mouse_controls_3d_label_list_container_layout = QVBoxLayout()
        self.mouse_controls_3d_label_list_container_layout.setContentsMargins(10, 0, 0, 0)
        self.mouse_controls_3d_label_list_container_layout.setSpacing(0)
        self.mouse_controls_3d_label_list_container.setLayout(self.mouse_controls_3d_label_list_container_layout)


        self.mouse_3d_image_widget = QLabel(self)
        self.mouse_3d_image_widget.setPixmap(QPixmap.fromImage(self.mouse_image.scaledToWidth(350, Qt.TransformationMode.SmoothTransformation)))

        self.mouse_3d_image_controls_container_layout.addWidget(self.mouse_3d_image_widget)
        self.mouse_3d_image_controls_container_layout.addWidget(self.mouse_controls_3d_label_list_container)

        self.mouse_3d_image_controls_container_layout.addWidget(self.mouse_controls_3d_label_list_container)
        self.mouse_3d_image_controls_container_layout.addStretch(1)

        self.mouse_3d_layout.addStretch(1)

        self.right_click_3d_label = QLabel(str(MouseAction.ADD_TO_SEGMENTATION) + " ((+shift) erase)")
        self.middle_click_3d_label = QLabel("(+shift) " + str(MouseAction.MOVE_SPHERE))
        self.scroll_3d_label = QLabel("(+shift) " + str(MouseAction.RESIZE_SPHERE))
        self.left_click_3d_label = QLabel('n/a')
        windows_spacings = [50, 12, 20, 26]
        mac_spacings = [50, 12, 20, 24]
        linux_spacings = [52, 14, 22, 26]
        if sys.platform == "win32":
            spacings = windows_spacings
        elif sys.platform == "darwin":
            spacings = mac_spacings
        else:
            spacings = linux_spacings
        for label, spacing in zip(
            [self.right_click_3d_label, self.middle_click_3d_label, self.scroll_3d_label, self.left_click_3d_label], spacings
        ):
            self.mouse_controls_3d_label_list_container_layout.addSpacing(spacing)
            self.mouse_controls_3d_label_list_container_layout.addWidget(label)
        self.mouse_controls_3d_label_list_container_layout.addStretch()
        self.tab_widget.addTab(self.mouse_3d_widget, "Mouse (3D)")

    def _add_vr_tab(self):
        self.vr_outer_widget = QWidget(self)
        self.vr_outer_layout = QHBoxLayout()
        self.vr_outer_widget.setLayout(self.vr_outer_layout)

        self.control_label_container = QWidget()
        self.control_label_container_layout = QVBoxLayout()
        self.control_label_container.setLayout(self.control_label_container_layout)

        self.vr_controller_container = QWidget()
        self.vr_controller_container_layout = QVBoxLayout()
        self.vr_controller_container.setLayout(self.vr_controller_container_layout)

        self.vr_controller_picture = QLabel(self)
        self.vr_controller_picture.setPixmap(QPixmap.fromImage(self.right_hand_image.scaledToWidth(350, Qt.TransformationMode.SmoothTransformation)))
        self.vr_controller_container_layout.addWidget(self.vr_controller_picture)
        self.vr_controller_container_layout.setContentsMargins(0, 0, 0, 0)
        self.vr_controller_container_layout.addStretch(1)
        self.vr_outer_layout.addWidget(self.vr_controller_container)
        self.vr_outer_layout.addWidget(self.control_label_container)
        self.vr_outer_layout.addStretch(1)
        self.thumbstick_label = QLabel(str(HandAction.RESIZE_CURSOR))
        self.menu_button_label = QLabel("n/a")
        self.trigger_label = QLabel("n/a")
        self.grip_label = QLabel(str(HandAction.MOVE_CURSOR))
        self.a_button_label = QLabel(str(HandAction.ADD_TO_SEGMENTATION))
        self.b_button_label = QLabel(str(HandAction.ERASE_FROM_SEGMENTATION))
        windows_spacings = [90, 50, 16, 20, 80, 40]
        mac_spacings = [90, 50, 16, 20, 80, 40]
        linux_spacings = [92, 52, 18, 22, 82, 42]
        if sys.platform == "win32":
            spacings = windows_spacings
        elif sys.platform == "darwin":
            spacings = mac_spacings
        else:
            spacings = linux_spacings
        for label, spacing in zip(
            [self.thumbstick_label, self.menu_button_label, self.trigger_label, self.grip_label, self.a_button_label, self.b_button_label]
            , spacings
        ):
            self.control_label_container_layout.addSpacing(spacing)
            self.control_label_container_layout.addWidget(label)
        self.control_label_container_layout.setContentsMargins(6, 0, 0, 0)
        self.control_label_container_layout.setSpacing(0)
        self.control_label_container_layout.addStretch()
        self.tab_widget.addTab(self.vr_outer_widget, "VR Controller")

class SegmentationTool(ToolInstance):
    # TODO: Sphere cursor for 2D, extend to VR

    help = "help:user/tools/segmentations.html"
    SESSION_ENDURING = True
    num_segmentations_created = 0

    def __init__(self, session = None, name = "Segmentations"):
        super().__init__(session, name)
        self.is_vr = False
        self._construct_ui()

    def _construct_ui(self):
        # Construct the GUI
        self.tool_window = MainToolWindow(self)
        self.settings = get_settings(self.session)
        self.parent = self.tool_window.ui_area
        self.controls_dialog = SegmentationToolControlsDialog(self.parent, self.session)
        self.main_layout = QVBoxLayout()
        self.active_seg: Volume | None = None

        self.view_dropdown_container = QWidget(self.parent)
        self.view_dropdown_layout = QHBoxLayout()
        self.view_dropdown_label = QLabel("View layout")
        self.view_dropdown = QComboBox(self.parent)
        for view in ViewMode:
            self.view_dropdown.addItem(str(view))
        self.view_dropdown.setCurrentIndex(self.settings.default_view)
        self.view_dropdown.currentIndexChanged.connect(self._on_view_changed)
        self.control_information_button = QToolButton()
        self.control_information_button.setMinimumWidth(1)
        self.control_information_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.control_information_button.setIcon(get_qt_icon("gear"))
        self.control_information_button.setToolTip("Segmentation Settings")
        self.control_information_button.clicked.connect(self.showControlsDialog)
        self.view_dropdown_layout.addWidget(self.view_dropdown_label)
        self.view_dropdown_layout.addWidget(self.view_dropdown, 1)
        self.view_dropdown_layout.addStretch(2)
        self.view_dropdown_layout.addWidget(self.control_information_button)
        self.view_dropdown_container.setLayout(self.view_dropdown_layout)
        self.old_mouse_bindings = {
            'left': {
                'none': None
                , 'shift': None
                , 'ctrl': None
                , 'command': None
                , 'alt': None
            }
            , 'right': {
                'none': None
                , 'shift': None
                , 'ctrl': None
                , 'command': None
                , 'alt': None
            }
            , 'middle': {
                'none': None
                , 'shift': None
                , 'ctrl': None
                , 'command': None
                , 'alt': None
            }
            , 'wheel': {
                'none': None
                , 'shift': None
                , 'ctrl': None
                , 'command': None
                , 'alt': None
            }
            , 'pause': {
                'none': None
                , 'shift': None
                , 'ctrl': None
                , 'command': None
                , 'alt': None
            }
        }
        self.old_hand_bindings = {
            'trigger': None,
            'grip': None,
            'touchpad': None,
            'thumbstick': None,
            'menu': None,
            'a': None,
            'b': None,
            'x': None,
            'y': None
        }
        self.view_dropdown_layout.setContentsMargins(0, 0, 0, 0)
        self.view_dropdown_layout.setSpacing(0)
        def _not_volume_surface_or_segmentation(m):
            ok_to_list = isinstance(m, DICOMVolume)
            # This will run over all models which may not have DICOM data...
            try:
                if hasattr(m.data, "dicom_data"):
                    ok_to_list &= bool(m.data.dicom_data) # SEGs have none
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

        self.mouse_modes_changed = False
        self.hand_modes_changed = False
        self.control_checkbox_container = QWidget()
        self.control_checkbox_layout = QHBoxLayout()

        self.guidelines_checkbox = QCheckBox("Plane guidelines")
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
        self.edit_seg_metadata_button.setVisible(False)
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

        self.intensity_range_checkbox = QCheckBox("Restrict segmentation to intensity range")
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
        self.main_layout.addWidget(self.intensity_range_checkbox)
        self.main_layout.addWidget(self.slider_container)

        self.main_layout.addStretch()
        self.main_layout.setContentsMargins(6, 0, 6, 6)
        self.main_layout.setSpacing(0)

        self.parent.setLayout(self.main_layout)
        self.tool_window.manage('side')
        self.segmentation_cursors = {}
        self.segmentation_sphere = None
        self.segmentations = {}
        self.current_segmentation = None
        self.reference_model = None
        self.threshold_max = 0
        self.threshold_min = 0

        self.model_added_handler = self.session.triggers.add_handler(ADD_MODELS, self._on_model_added_to_session)
        self.model_closed_handler = self.session.triggers.add_handler(REMOVE_MODELS, self._on_model_removed_from_session)

        # Keep track of the last layout used so we know whether to add new segmentation
        # overlays to views when the layout changes
        self.previous_layout = None
        self.current_layout = self.settings.default_view

        # TODO: VR started trigger
        if not self.session.ui.main_window.view_layout == "orthoplanes":
            if self.settings.default_view == ViewMode.TWO_BY_TWO:
                self._create_2d_segmentation_pucks()
                run(self.session, "dicom view fourup")
            elif self.settings.default_view == ViewMode.ORTHOPLANES_OVER_3D:
                self._create_2d_segmentation_pucks()
                run(self.session, "dicom view overunder")
            elif self.settings.default_view == ViewMode.ORTHOPLANES_BESIDE_3D:
                self._create_2d_segmentation_pucks()
                run(self.session, "dicom view sidebyside")
            elif self.settings.default_view == ViewMode.DEFAULT_DESKTOP:
                self._create_3d_segmentation_sphere()
                if self.settings.set_mouse_modes_automatically:
                    self._set_3d_mouse_modes()
            else:
                self._create_3d_segmentation_sphere()
                if self.settings.start_vr_automatically:
                    self._start_vr()

        # Do the initial population of the segmentation list
        for model in self.session.models:
            if type(model) is DICOMVolume and model.is_segmentation():
                self.segmentation_list.addItem(SegmentationListItem(parent = self.segmentation_list, segmentation = model))
                if self.session.ui.main_window.view_layout == "orthoplanes":
                    self.session.ui.main_window.main_view.add_segmentation(model)
        self.segmentations_by_model = {}
        self.tool_window.fill_context_menu = self.fill_context_menu
        self._surface_chosen()

    def fill_context_menu(self, menu, x, y):
        show_settings_action = QAction("Settings...", menu)
        show_settings_action.triggered.connect(self.showControlsDialog)
        menu.addAction(show_settings_action)

    def showControlsDialog(self):
        self.controls_dialog.show()

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

    def _on_model_removed_from_session(self, *args):
        # If this model is a DICOM segmentation, add it to the list of segmentations
        _, model_list = args
        if model_list:
            for model in model_list:
                for row in range(self.segmentation_list.count()):
                    item = self.segmentation_list.item(row)
                    if item.segmentation == model:
                        seg_item = self.segmentation_list.takeItem(row)
                        segments = [seg_item.segmentation]
                        seg_item.segmentation = None
                        del seg_item
                        if self.session.ui.main_window.view_layout == "orthoplanes":
                            self.session.ui.main_window.main_view.remove_segmentation(model)
                        break

    def delete(self):
        self.session.triggers.remove_handler(self.model_added_handler)
        self.session.triggers.remove_handler(self.model_closed_handler)
        # TODO: Restore old mouse modes if necessary
        if self.session.ui.main_window.view_layout == "orthoplanes":
            self.session.ui.main_window.main_view.clear_segmentation_tool()
        # When a session is closed, models are deleted before tools, so we need to
        # fail gracefully if the models have already been deleted
        try:
            self._destroy_2d_segmentation_pucks()
            self._destroy_3d_segmentation_sphere()
        except TypeError:
            pass
        self._reset_3d_mouse_modes()
        self._reset_vr_hand_modes()
        super().delete()

    def _set_3d_mouse_modes(self):
        for binding in self.session.ui.mouse_modes.bindings:
            if not binding.modifiers:
                self.old_mouse_bindings[binding.button]['none'] = binding.mode.name
            else:
                for modifier in binding.modifiers:
                    self.old_mouse_bindings[binding.button][modifier] = binding.mode.name
        run(self.session, "ui mousemode shift wheel 'resize segmentation cursor'")
        run(self.session, "ui mousemode right 'create segmentations'")
        run(self.session, "ui mousemode shift right 'erase segmentations'")
        run(self.session, "ui mousemode shift middle 'move segmentation cursor'")
        self.mouse_modes_changed = True

    def _reset_3d_mouse_modes(self):
        """Set mouse modes back to what they were but only if we changed them automatically.
        If you set the mode by hand, or in between the change and restore you're on your own!"""
        if self.mouse_modes_changed:
            run(self.session, "ui mousemode shift wheel '" + self.old_mouse_bindings['wheel']['shift'] + "'" if self.old_mouse_bindings['wheel']['shift'] else "ui mousemode shift wheel 'none'")
            run(self.session, "ui mousemode right '" + self.old_mouse_bindings['right']['none'] + "'" if self.old_mouse_bindings['right']['none'] else "ui mousemode right 'none'")
            run(self.session, "ui mousemode shift right '" + self.old_mouse_bindings['right']['shift'] + "'" if self.old_mouse_bindings['right']['shift'] else "ui mousemode shift right 'none'")
            run(self.session, "ui mousemode shift middle '" + self.old_mouse_bindings['middle']['shift'] + "'" if self.old_mouse_bindings['middle']['shift'] else "ui mousemode shift middle 'none'")
        self.mouse_modes_changed = False

    def _set_vr_hand_modes(self):
        self.hand_modes_changed = True
        if type(self.session.main_view.camera) is SteamVRCamera:
            vr_camera = steamvr_camera
            vr_button = steamvr_button
            from openvr import \
                k_EButton_Grip as grip, \
                k_EButton_ApplicationMenu as menu, \
                k_EButton_SteamVR_Trigger as trigger, \
                k_EButton_SteamVR_Touchpad as touchpad, \
                k_EButton_A as a
            button_names = { grip: 'grip', menu: 'menu', trigger: 'trigger', touchpad: 'thumbstick', a: 'a' }
            c = vr_camera(self.session)
            hclist = [hc for hc in c.hand_controllers() if hc._side == str(self.settings.vr_handedness).lower()]
            if not hclist:
                ... # error
            hc = hclist[0]
            for button, binding in hc._modes.items():
                self.old_hand_bindings[button_names[button]] = binding.name
        elif type(self.session.main_view.camera) is OpenXRCamera:
            # TODO
            vr_camera = openxr_camera
            vr_button = openxr_button
        run(self.session, f'vr button b \'erase segmentations\' hand { str(self.settings.vr_handedness).lower() }')
        run(self.session, f'vr button a \'create segmentations\' hand { str(self.settings.vr_handedness).lower() }')
        run(self.session, f'vr button x \'toggle segmentation visibility\' hand left')
        run(self.session, f'vr button thumbstick \'resize segmentation cursor\' hand { str(self.settings.vr_handedness).lower() }')
        run(self.session, f'vr button grip \'move segmentation cursor\' hand { str(self.settings.vr_handedness).lower() }')

    def _reset_vr_hand_modes(self):
        """Set hand modes back to what they were but only if we changed them automatically.
        If you set the mode by hand, or in between the change and restore you're on your own!"""
        if self.hand_modes_changed:
            run(self.session, f'vr button trigger {self.old_hand_modes["trigger"]}')
            run(self.session, f'vr button thumbstick {self.old_hand_modes["thumbstick"]}')
            run(self.session, f'vr button grip {self.old_hand_modes["grip"]}')
            run(self.session, f'vr button a {self.old_hand_modes["a"]}')
            run(self.session, f'vr button b {self.old_hand_modes["b"]}')
            run(self.session, f'vr button x {self.old_hand_modes["x"]}')
        self.hand_modes_changed = False

    def _start_vr(self):
        run(self.session, "vr on")
        if self.settings.set_hand_modes_automatically:
            self._set_vr_hand_modes()

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
        # TODO: Set the height to the thickness in that direction
        self.segmentation_cursors = {
            Axis.AXIAL:    SegmentationDisk(self.session, Axis.AXIAL, height=5)
            , Axis.CORONAL:  SegmentationDisk(self.session, Axis.CORONAL, height=5)
            , Axis.SAGITTAL: SegmentationDisk(self.session, Axis.SAGITTAL, height=5)
        }
        for cursor in self.segmentation_cursors.values():
            cursor.display = initial_display
        self.session.models.add(self.segmentation_cursors.values())

    def _destroy_2d_segmentation_pucks(self) -> None:
        self.session.models.remove(self.segmentation_cursors.values())
        self.segmentation_cursors = {}

    def _create_3d_segmentation_sphere(self) -> None:
        self.segmentation_sphere = SegmentationSphere(
            "Segmentation Sphere"
            , self.session
        )
        if self.reference_model:
            self.segmentation_sphere.position = Place(origin = self.reference_model.bounds().center())

    def _destroy_3d_segmentation_sphere(self) -> None:
        if self.segmentation_sphere:
            self.session.models.remove([self.segmentation_sphere])
            self.segmentation_sphere = None

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
        if self.intensity_range_checkbox.isChecked() and value != 0:
            self.active_seg.set_2d_segment_data(axis, slice, positions, value, self.threshold_min, self.threshold_max)
        else:
            self.active_seg.set_2d_segment_data(axis, slice, positions, value)

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
        new_seg.set_parameters(surface_levels=[0.501])
        new_seg.set_step(1)
        new_seg.set_transparency(int((self.settings.default_segmentation_opacity / 100) * 255))
        num_items = self.segmentation_list.count()
        self.segmentation_list.setCurrentItem(self.segmentation_list.item(num_items - 1))

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
        if self.active_seg:
            self.active_seg.active = True
        if self.session.ui.main_window.view_layout == "orthoplanes":
            self.session.ui.main_window.main_view.redraw_all()

    def hide_active_segmentation(self):
        self.active_seg.display = False

    def show_active_segmentation(self):
        self.active_seg.display = True

    def _on_active_segmentation_changed(self, new, prev):
        if new:
            self.edit_seg_metadata_button.setEnabled(True)
            self.setActiveSegment(new.segmentation)
        else:
            self.edit_seg_metadata_button.setEnabled(False)
            self.setActiveSegment(None)

    def set_segmentation_step(self, step):
        if not self.active_seg:
            self.addSegment()
        self.active_seg.set_step(step)

    def _on_view_changed(self):
        self.previous_layout = self.current_layout
        self.current_layout = self.view_dropdown.currentIndex()
        need_to_register = False
        if self.view_dropdown.currentIndex() == ViewMode.TWO_BY_TWO:
            if self.is_vr:
                self.is_vr = False
                run(self.session, "vr off")
            self._reset_3d_mouse_modes()
            if self.segmentation_sphere:
                self._destroy_3d_segmentation_sphere()
            if not self.segmentation_cursors:
                self._create_2d_segmentation_pucks()
            run(self.session, "dicom view fourup")
            need_to_register = True
        elif self.view_dropdown.currentIndex() == ViewMode.ORTHOPLANES_OVER_3D:
            if self.is_vr:
                self.is_vr = False
                run(self.session, "vr off")
            self._reset_3d_mouse_modes()
            if self.segmentation_sphere:
                self._destroy_3d_segmentation_sphere()
            if not self.segmentation_cursors:
                self._create_2d_segmentation_pucks()
            run(self.session, "dicom view overunder")
            need_to_register = True
        elif self.view_dropdown.currentIndex() == ViewMode.ORTHOPLANES_BESIDE_3D:
            if self.is_vr:
                self.is_vr = False
                run(self.session, "vr off")
            self._reset_3d_mouse_modes()
            if self.segmentation_sphere:
                self._destroy_3d_segmentation_sphere()
            if not self.segmentation_cursors:
                self._create_2d_segmentation_pucks()
            run(self.session, "dicom view sidebyside")
            need_to_register = True
        else:
            if self.view_dropdown.currentIndex() == ViewMode.DEFAULT_VR:
                self.is_vr = True
            else:
                self.is_vr = False
            run(self.session, "dicom view default")
            if self.segmentation_cursors:
                self._destroy_2d_segmentation_pucks()
            if not self.segmentation_sphere:
                self._create_3d_segmentation_sphere()
            #if self.autostart_vr_checkbox.isChecked():
            #    run(self.session, "vr on")
            if self.view_dropdown.currentIndex() == ViewMode.DEFAULT_DESKTOP:
                if self.settings.set_mouse_modes_automatically:
                    self._set_3d_mouse_modes()
            if self.view_dropdown.currentIndex() == ViewMode.DEFAULT_VR:
                if self.settings.start_vr_automatically:
                    self._start_vr()
        if need_to_register:
            if not self.segmentation_cursors:
                self._create_2d_segmentation_pucks()
            if self.session.ui.main_window.view_layout == "orthoplanes":
                # If no models are open we will not successfully change the view, so
                # we need to check the view layout before continuing!
                self.session.ui.main_window.main_view.register_segmentation_tool(self)
                if self.guidelines_checkbox.isChecked():
                    self.session.ui.main_window.main_view.toggle_guidelines()
            if self.previous_layout not in {ViewMode.ORTHOPLANES_BESIDE_3D, ViewMode.ORTHOPLANES_OVER_3D, ViewMode.TWO_BY_TWO}:
                for i in range(self.segmentation_list.count()):
                    self.session.ui.main_window.main_view.add_segmentation(self.segmentation_list.item(i).segmentation)

    def set_view_dropdown(self, layout):
        if layout == "default":
            if self.is_vr:
                self.view_dropdown.setCurrentIndex(ViewMode.DEFAULT_VR)
            else:
                self.view_dropdown.setCurrentIndex(ViewMode.DEFAULT_DESKTOP)
        elif layout == "sidebyside":
            self.view_dropdown.setCurrentIndex(ViewMode.ORTHOPLANES_BESIDE_3D)
        elif layout == "overunder":
            self.view_dropdown.setCurrentIndex(ViewMode.ORTHOPLANES_OVER_3D)
        else:
            self.view_dropdown.setCurrentIndex(ViewMode.TWO_BY_TWO)

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

    def move_sphere(self, delta_xyz):
        sm = self.segmentation_sphere
        dxyz = sm.scene_position.inverse().transform_vector(delta_xyz)	# Transform to sphere local coords.
        sm.position = sm.position * translation(dxyz)

    def setSphereRegionToValue(self, origin, radius, value=1):
        if not self.active_seg:
            self.addSegment()
        if self.intensity_range_checkbox.isChecked() and value != 0:
            self.active_seg.set_sphere_data(origin, radius, value, self.threshold_min, self.threshold_max)
        else:
            self.active_seg.set_sphere_data(origin, radius, value)

    def addSphereToSegment(self, origin, radius):
        self.setSphereRegionToValue(origin, radius, 1)

    def removeSphereFromSegment(self, origin, radius):
        self.setSphereRegionToValue(origin, radius, 0)
