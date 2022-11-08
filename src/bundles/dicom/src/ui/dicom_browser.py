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

from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QHeaderView
    , QWidget, QLabel, QAbstractItemView
    , QPushButton
)

from chimerax.core.models import (
    ADD_MODELS, REMOVE_MODELS
    , MODEL_ID_CHANGED, MODEL_NAME_CHANGED
)

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow

from .widgets import DICOMTable
from . import DICOMMetadata

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
        _logger = UI.instance().session.logger
        _session = UI.instance().session
    except (NameError, AttributeError):
        # We didn't have either of ChimeraX's UIs, or they were uninitialized.
        # We're either in some other application or being used as a library.
        # Default to passed in sessions and the Python logging module
        import logging
        _session = None
        _logger = logging.getLogger()
        _logger.status = _logger.info

# TODO: Make singleton
class DICOMBrowserTool(ToolInstance):
    def __init__(self, session = None, name = "DICOM Browser"):
        """Bring up a tool to explore DICOM models open in the session."""
        session = session or _session
        super().__init__(session, name)

        # Construct the GUI
        self.tool_window = MainToolWindow(self)
        self.parent = self.tool_window.ui_area
        self.main_layout = QVBoxLayout()

        self.buttons_label = QLabel("For Highlighted Entries:", parent=self.parent)
        self.load_buttons_widget = QWidget(self.parent)
        self.load_button_container = QHBoxLayout()
        self.load_button_container.addWidget(self.buttons_label)

        self.load_pt_button = QPushButton("Load Patient Metadata", parent=self.load_buttons_widget)
        self.load_st_button = QPushButton("Load Study Metadata", parent=self.load_buttons_widget)
        self.load_sr_button = QPushButton("Load Series Metadata", parent=self.load_buttons_widget)
        self.load_button_container.addWidget(self.load_pt_button)
        self.load_button_container.addWidget(self.load_st_button)
        self.load_button_container.addWidget(self.load_sr_button)
        # self.load_pt_button.clicked.connect(lambda: self.patient_metadata(self.patient_table.selected))
        # self.load_st_button.clicked.connect(lambda: self.study_metadata(self.study_table.selected))
        # self.load_sr_button.clicked.connect(lambda: self.series_metadata(self.series_table.selected))

        self.load_buttons_widget.setLayout(self.load_button_container)

        # self.patient_control_widget = QWidget(self.parent)
        # self.study_control_widget = QWidget(self.parent)
        # self.series_control_widget = QWidget(self.parent)
        # self.patient_control_widget.setVisible(False)
        # self.study_control_widget.setVisible(False)
        # self.series_control_widget.setVisible(False)

        # self.patient_table = DICOMTable(self.patient_control_widget, None, self.parent)
        # self.study_table = DICOMTable(self.study_control_widget, None, self.parent)
        # self.series_table = DICOMTable(self.series_control_widget, None, self.parent)

        self.patient_label = QLabel("Patients")
        self.study_label = QLabel("Studies")
        self.series_label = QLabel("Series")

        # Set the selected patient to whichever happens to be first
        # self.patient_table.data = self.models
        # self.study_table.data = list(chain.from_iterable([x.studies for x in chain(self.models)]))
        # self.series_table.data = list(chain.from_iterable([x.series for x in self.study_table.data]))

        # self._set_table_columns()

        # self.patient_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.study_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.series_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        #self.study_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        #self.series_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # self.patient_table.get_selection.connect(self.on_patient_highlighted)

        self.main_layout.addWidget(self.patient_label)
        # self.main_layout.addWidget(self.patient_table)
        self.main_layout.addWidget(self.study_label)
        # self.main_layout.addWidget(self.study_table)
        self.main_layout.addWidget(self.series_label)
        # self.main_layout.addWidget(self.series_table)

        self.main_layout.addWidget(self.load_buttons_widget)

        self.tool_window.ui_area.setLayout(self.main_layout)
        # self.patient_table.launch()
        # self.study_table.launch()
        # self.series_table.launch()
        session.triggers.add_handler(
            ADD_MODELS, lambda *args: self._see_what_happened(*args)
        )
        # session.triggers.add_handler(REMOVE_MODELS,
        #    lambda *args: self._initiate_fill_tree(*args, always_rebuild=True))
        # session.triggers.add_handler(MODEL_ID_CHANGED,
        #    lambda *args: self._initiate_fill_tree(*args, always_rebuild=True, countdown=3))
        # session.triggers.add_handler(MODEL_NAME_CHANGED,
        #    lambda *args: self._initiate_fill_tree(*args, simple_change=True, countdown=3))

        self.tool_window.manage()

    def patient_metadata(self, selection):
        if len(selection) > 0:
            return DICOMMetadata.from_patients(self.session, selection)

    def study_metadata(self, selection):
        if len(selection) > 0:
            return DICOMMetadata.from_studies(self.session, selection)

    def series_metadata(self, selection):
        if len(selection) > 0:
            return DICOMMetadata.from_series(self.session, selection)

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

    def _check_new_dicoms(self, _, models):
        # DICOM models will always come in on a patient-wise basis, so only
        # check the type of the first one
        if type(models[0]) is Patient:
            self.add_patient(models)

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
