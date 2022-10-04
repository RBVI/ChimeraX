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
from Qt.QtWidgets import QVBoxLayout, QHBoxLayout
from Qt.QtWidgets import QWidget
from Qt.QtWidgets import QLabel
from Qt.QtWidgets import QAbstractItemView

from chimerax.core.tools import ToolInstance
from chimerax.help_viewer import show_url
from chimerax.ui import MainToolWindow
from pydicom.multival import MultiValue

from .widgets import DICOMTable

dicom_template_url = "http://dicomlookup.com/lookup.asp?sw=Tnumber&q=%s" # noqa they don't have https

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
    def __init__(self, session):
        """Bring up a tool to view fine-grained metadata in DICOM files.
        session: A ChimeraX session
        dicom_file: The data structure returned by pydicom.dcmread()
        """
        self.files = []
        self.display_name = "DICOM Metadata"
        super().__init__(session, self.display_name)

    def build_ui(self):
        # Construct the GUI
        self.tool_window = MainToolWindow(self)
        self.parent = self.tool_window.ui_area
        self.main_layout = QVBoxLayout()
        self.upper_widget = QWidget(self.parent)
        self.upper_layout = QHBoxLayout()
        self.filename_label = QLabel()
        self.table_control_widget = QWidget(self.parent)
        self.table_control_widget.setVisible(False)
        self.table = DICOMTable(self.table_control_widget, None, self.parent)
        self.table.data = [MetadataRow(item) for item in iter(self.files[0])]
        self._set_table_columns()
        self.table.sortByColumn(0, Qt.AscendingOrder)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.main_layout.addWidget(self.upper_widget)
        self.main_layout.addWidget(self.table)
        self.tool_window.ui_area.setLayout(self.main_layout)
        self.table.get_selection.connect(self.open_dicom_webpage)
        self.table.launch()

        self.tool_window.fill_context_menu = self.fill_context_menu

        self.tool_window.manage()

    @classmethod
    def from_series(cls, series):
        c = cls(series.session)
        c.add_series(series)
        c.build_ui()
        return c

    def add_dicom_file(self, file):
        self.files.append(file)

    def fill_context_menu(self, menu, x, y):
        pass

    def add_series(self, series):
        self.files.extend(series.files)

    def on_file_changed(self):
        pass

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
