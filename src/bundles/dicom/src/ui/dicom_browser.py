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
from itertools import chain


from Qt.QtGui import QAction
from Qt.QtWidgets import QVBoxLayout
from Qt.QtWidgets import QHeaderView
from Qt.QtWidgets import QWidget
from Qt.QtWidgets import QLabel
from Qt.QtWidgets import QAbstractItemView

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow

from .widgets import DICOMTable

# TODO: Make singleton
class DICOMBrowserTool(ToolInstance):
    def __init__(self, models=None):
        """Bring up a tool to explore DICOM models open in the session.
        session: A ChimeraX session
        model: The root model containing patients, series, and files underneath
        """
        if models:
            self.models = models if type(models) is list else [models]
            session = self.models[0].session
        else:
            self.models = None
            session = None
        self.display_name = "DICOM Browser"
        super().__init__(session, self.display_name)

        # Construct the GUI
        self.tool_window = MainToolWindow(self)
        self.parent = self.tool_window.ui_area
        self.main_layout = QVBoxLayout()

        self.patient_control_widget = QWidget(self.parent)
        self.study_control_widget = QWidget(self.parent)
        self.series_control_widget = QWidget(self.parent)
        self.patient_control_widget.setVisible(False)
        self.study_control_widget.setVisible(False)
        self.series_control_widget.setVisible(False)

        self.patient_table = DICOMTable(self.patient_control_widget, None, self.parent)
        self.study_table = DICOMTable(self.study_control_widget, None, self.parent)
        self.series_table = DICOMTable(self.series_control_widget, None, self.parent)

        self.patient_label = QLabel("Patients")
        self.study_label = QLabel("Studies")
        self.series_label = QLabel("Series")

        # Set the selected patient to whichever happens to be first
        self.patient_table.data = self.models
        self.study_table.data = list(chain.from_iterable([x.studies for x in chain(self.models)]))
        self.series_table.data = list(chain.from_iterable([x.series for x in self.study_table.data]))

        self._set_table_columns()

        self.patient_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.study_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.series_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        #self.study_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        #self.series_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.patient_table.get_selection.connect(self.on_patient_highlighted)

        self.tool_window.fill_context_menu = self.fill_context_menu

        self.main_layout.addWidget(self.patient_label)
        self.main_layout.addWidget(self.patient_table)
        self.main_layout.addWidget(self.study_label)
        self.main_layout.addWidget(self.study_table)
        self.main_layout.addWidget(self.series_label)
        self.main_layout.addWidget(self.series_table)

        self.tool_window.ui_area.setLayout(self.main_layout)
        self.patient_table.launch()
        self.study_table.launch()
        self.series_table.launch()

        self.tool_window.fill_context_menu = self.fill_context_menu

        self.tool_window.manage()

    def fill_context_menu(self, menu, x, y):
        open_metadata_action = QAction("Load Metadata", menu)
        open_metadata_action.triggered.connect(lambda: self.metadata(self.patient_table.selected))
        menu.addAction(open_metadata_action)

    def metadata(self, selection):
        pass

    def add_patient(self, patients):
        if not patients:
            return
        else:
            try:
                for patient in patients:
                    self.models[patient.uid] = patient
            except TypeError:
                self.models[patients.uid] = patients

    def on_patient_highlighted(self):
        pass

    def on_series_highlighted(self):
        pass

    def _set_table_columns(self):
        if not self.patient_table:
            raise RuntimeError("Cannot set columns for non-existent patient table.")
        if not self.study_table:
            raise RuntimeError("Cannot set columns for non-existent study table.")
        if not self.series_table:
            raise RuntimeError("Cannot set columns for non-existent series table.")

        self.patient_table.add_column("Patient Name", lambda x: x.patient_name)
        self.patient_table.add_column("Patient ID", lambda x: x.pid)
        self.patient_table.add_column("Birth Date", lambda x: _format_date(x.birth_date_as_datetime))
        self.patient_table.add_column("Sex", lambda x: x.patient_sex)
        self.patient_table.add_column("Studies", lambda x: len(x.studies))
        self.patient_table.add_column("Most Recent Study", lambda x: len(x.studies))
        self.study_table.add_column("Date", lambda x: _format_date(x.date_as_datetime))
        self.study_table.add_column("ID", lambda x: x.study_id)
        self.study_table.add_column("Description", lambda x: x.description)
        self.study_table.add_column("Series", lambda x: len(x.series))
        self.series_table.add_column("Series #", lambda x: x.number)
        self.series_table.add_column("Description", lambda x: x.description)
        self.series_table.add_column("Modality", lambda x: x.modality)
        self.series_table.add_column("Size", lambda x: x.size)
        self.series_table.add_column("Count", lambda x:  len(x.files))

def _format_date(date):
    if date:
        return date.strftime("%Y-%m-%d")
    else:
        return ""
