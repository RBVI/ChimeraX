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
from Qt.QtCore import Qt
from Qt.QtGui import QAction

from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QSlider
    , QSpinBox, QWidget, QLabel
    , QAbstractItemView, QSizePolicy
)

from chimerax.core.tools import ToolInstance
from chimerax.help_viewer import show_url
from chimerax.ui import MainToolWindow
from pydicom.multival import MultiValue

from .widgets import DICOMTable

dicom_template_url: str = "http://dicomlookup.com/lookup.asp?sw=Tnumber&q=%s" # noqa they don't have https

try:
    # We are inside GUI ChimeraX
    from chimerax.ui.gui import UI
except (ModuleNotFoundError, ImportError):
    # We could be in NoGUI ChimeraX
    try:
        from chimerax.core.nogui import UI
    except (ModuleNotFoundError, ImportError):
        pass
finally:
    try:
        _session = UI.instance().session
    except (NameError, AttributeError):
        # We didn't have either of ChimeraX's UIs, or they were uninitialized.
        # We're either in some other application or being used as a library.
        # Default to passed in sessions.
        _session = None


class MetadataRow:
    """Takes in and stores a dictionary. This class only exists to coerce Python into hashing a dictionary."""

    # Save on memory by suppressing the internal class dictionary.
    # Only allocate these slots.
    __slots__ = ['_internal_dict']

    def __init__(self, row: dict):
        self._internal_dict = row

    def __getattr__(self, key):
        return getattr(self._internal_dict, key)


class DICOMMetadata(ToolInstance):
    def __init__(self, session = None, name = "DICOM Metadata"):
        """Bring up a tool to view fine-grained metadata in DICOM files.
        session: A ChimeraX session
        dicom_file: The data structure returned by pydicom.dcmread()
        """
        self.files = []
        session = session or _session
        super().__init__(session, name)

    def build_ui(self):
        # Construct the GUI
        self.tool_window = MainToolWindow(self)
        self.parent = self.tool_window.ui_area
        self.main_layout = QVBoxLayout()
        self.table_control_widget = QWidget(self.parent)
        self.table_control_widget.setVisible(False)
        self.table = DICOMTable(self.table_control_widget, None, self.parent)
        self.table.data = [MetadataRow(item) for item in iter(self.files[0])]
        self._set_table_columns()
        self.table.sortByColumn(0, Qt.AscendingOrder)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.controls_container = QWidget(parent=self.parent)
        self.controls_layout = QHBoxLayout()
        self.file_box = QSpinBox(parent=self.parent)
        self.file_slider = QSlider(Qt.Orientation.Horizontal, parent=self.parent)

        self.file_slider.setMinimum(1)
        self.file_slider.setMaximum(len(self.files))
        self.file_box.setMinimum(1)
        self.file_box.setMaximum(len(self.files))

        self.file_path_label = QLabel(parent=self.parent)
        self.file_path_label.setMargin(6)
        self.file_path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_path_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.file_path_label.setText("File path: %s" % self.files[0].path)

        self.file_slider.valueChanged[int].connect(self.on_file_changed_slider)
        self.file_box.valueChanged[int].connect(self.on_file_changed_box)

        self.controls_layout.addWidget(self.file_slider)
        self.controls_layout.addWidget(self.file_box)

        self.controls_container.setLayout(self.controls_layout)
        self.main_layout.addWidget(self.file_path_label)
        self.main_layout.addWidget(self.controls_container)
        self.main_layout.addWidget(self.table)
        self.tool_window.ui_area.setLayout(self.main_layout)
        self.table.get_selection.connect(self.open_dicom_webpage)
        self.table.launch()

        self.tool_window.fill_context_menu = self.fill_context_menu

        self.tool_window.manage()

    @classmethod
    def from_patients(cls, session, patient):
        c = cls(session)
        c.add_patients(patient)
        c.build_ui()
        return c

    @classmethod
    def from_studies(cls, session, study):
        c = cls(session)
        c.add_studies(study)
        c.build_ui()
        return c

    @classmethod
    def from_series(cls, session, series):
        c = cls(session)
        c.add_series(series)
        c.build_ui()
        return c

    def add_patients(self, patients) -> None:
        for patient in patients:
            self.add_patient(patient)

    def add_patient(self, patient) -> None:
        for study in patient:
            self.add_study(study)

    def add_studies(self, studies) -> None:
        for study in studies:
            self.add_study(study)

    def add_study(self, study) -> None:
        for series in study:
            self.add_series(series)

    def add_series(self, series) -> None:
        if type(series) is list:
            for series_ in series:
                self.add_series(series_)
        else:
            for file in series.files:
                self.add_dicom_file(file)

    def add_dicom_file(self, file) -> None:
        self.files.append(file)

    def fill_context_menu(self, menu, x, y) -> None:
        pass

    def _on_file_changed(self, value: int) -> None:
        file = self.files[value - 1]
        self.table.data = [MetadataRow(item) for item in iter(file)]
        self.file_path_label.setText(f'File path: {str(file.path)}')

    def on_file_changed_slider(self, value: int) -> None:
        self._on_file_changed(value)
        self.file_box.setValue(value)

    def on_file_changed_box(self, value: int) -> None:
        self._on_file_changed(value)
        self.file_slider.setValue(value)

    def num_files(self):
        return len(self.files)

    def _set_table_columns(self):
        if not self.table:
            raise RuntimeError("Cannot set columns for non-existent metadata table.")
        self.table.add_column("Tag", lambda x: str(x.tag).upper())
        self.table.add_column("Attribute", lambda x: x.description())
        self.table.add_column("Value", lambda x: _format_multivalue(x))
        self.table.add_column("VR", lambda x: x.VR)

    def open_dicom_webpage(self, selection):
        url = dicom_template_url % str(selection[0].tag).replace(" ", "")
        show_url(self.session, url)

def _format_multivalue(element):
    if type(element.value) is MultiValue:
        return ",".join([str(v).strip("'") for v in element.value])
    else:
        return element.repval.strip("'")
