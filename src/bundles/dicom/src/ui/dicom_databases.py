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
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QHeaderView
    , QWidget, QLabel, QAbstractItemView
    , QPushButton, QAction, QComboBox
    , QStackedWidget, QSizePolicy
)

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run

from ..dicom_fetch import *
from .widgets import DICOMTable

class DICOMDatabases(ToolInstance):

    help = "help:user/tools/dicomdatabases.html"

    def __init__(self, session = None, name = "Download DICOM"):
        """Bring up a tool to explore DICOM models open in the session."""
        super().__init__(session, name)

        self._main_table_data = []

        # Construct the GUI
        self.tool_window = MainToolWindow(self, close_destroys=False)
        self.parent = self.tool_window.ui_area
        self.main_layout = QVBoxLayout()

        self.interface_stack = QStackedWidget(self.parent)
        self.main_layout.addWidget(self.interface_stack)

        self.database_entries_container = QWidget(self.interface_stack)
        self.database_entries_layout = QVBoxLayout(self.database_entries_container)
        self.database_entries_container.setLayout(self.database_entries_layout)

        self.available_dbs = QComboBox(self.database_entries_container)
        self.database_entries_control_widget = QWidget(self.database_entries_container)
        self.database_label = QLabel("Database:", self.database_entries_container)
        self.control_container = QWidget(self.database_entries_container)
        self.control_layout = QHBoxLayout(self.control_container)

        self.control_container.setLayout(self.control_layout)

        self.control_layout.addWidget(self.database_label)
        self.control_layout.addWidget(self.available_dbs)
        self.control_layout.addStretch()
        self.dataset_highlighted_label = QLabel("For highlighted entries:")
        self.refine_dataset_button = QPushButton("Drill Down to Studies")
        self.control_layout.addWidget(self.dataset_highlighted_label)
        self.control_layout.addWidget(self.refine_dataset_button)
        self.database_entries_control_widget.setVisible(False)
        self.database_entries = DICOMTable(self.database_entries_control_widget, None, self.database_entries_container)
        self.database_entries_layout.addWidget(self.database_entries)
        self.database_entries_layout.addWidget(self.database_entries_control_widget)
        self.database_entries_layout.addWidget(self.control_container)
        self.database_entries_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setContentsMargins(4,4,4,4)


        self.database_entries.add_column("Dataset", lambda x: x.collection)
        self.database_entries.add_column("Number of Patients", lambda x: x.count)

        self.interface_stack.addWidget(self.database_entries_container)

        self.combo_box_model = self.available_dbs.model()
        self.available_dbs.addItem("TCIA")
        self.database_entries.launch()
        self.database_entries.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.interface_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.database_entries.get_selection.connect(self._on_main_table_double_clicked)

        self.study_entries_container = QWidget(self.interface_stack)
        self.study_entries_layout = QVBoxLayout(self.study_entries_container)
        self.back_to_search_button = QPushButton("Back to Collections")
        self.study_highlighted_label = QLabel("For highlighted entries:")
        self.refine_study_button = QPushButton("Drill Down to Series")

        self.study_entries_control_widget = QWidget(self.study_entries_container)
        self.study_entries_control_widget.setVisible(False)
        self.study_entries = DICOMTable(self.study_entries_control_widget, None, self.study_entries_container)
        self.study_entries.add_column("Study Instance UID", lambda x: x.suid)
        self.study_entries.add_column("Date", lambda x: x.date)
        self.study_entries.add_column("Description", lambda x: x.desc)
        self.study_entries.add_column("Patient ID", lambda x: x.patientId)
        self.study_entries.add_column("Patient Sex", lambda x: x.patientSex)
        self.study_entries.add_column("Patient Name", lambda x: x.patientName)
        self.study_entries.add_column("Series Count", lambda x: x.scount)
        self.study_entries_layout.addWidget(self.study_entries)
        self.study_view_control_container = QWidget(self.study_entries_container)
        self.study_view_control_layout = QHBoxLayout(self.study_view_control_container)
        self.study_view_control_layout.addWidget(self.back_to_search_button)
        self.study_view_control_layout.addStretch()
        self.study_view_control_layout.addWidget(self.study_highlighted_label)
        self.study_view_control_layout.addWidget(self.refine_study_button)
        self.study_entries_layout.addWidget(self.study_view_control_container)
        self.study_entries.get_selection.connect(self._on_study_table_double_clicked)
        self.study_entries.launch()
        self.back_to_search_button.clicked.connect(lambda: self._on_back_to_search_button_clicked())
        self.interface_stack.addWidget(self.study_entries_container)

        self.series_entries_container = QWidget(self.interface_stack)
        self.series_entries_layout = QVBoxLayout(self.series_entries_container)
        self.back_to_studies_button = QPushButton("Back to Studies")
        self.back_to_beginning_button = QPushButton("Back to Collections")
        self.series_highlighted_label = QLabel("For highlighted entries:")
        self.open_button = QPushButton("Download and Open")

        self.series_entries_control_widget = QWidget(self.series_entries_container)
        self.series_entries_control_widget.setVisible(False)
        self.series_entries = DICOMTable(self.series_entries_control_widget, None, self.series_entries_container)
        self.series_entries.add_column("Series Instance UID", lambda x: x.serUid)
        self.series_entries.add_column("Modality", lambda x: x.modality)
        self.series_entries.add_column("Protocol Name", lambda x: x.protocolName)
        self.series_entries.add_column("Series Description", lambda x: x.seriesDescription)
        self.series_entries.add_column("Body Part Examined", lambda x: x.bodyPart)
        self.series_entries.add_column("Patient ID", lambda x: x.patientID)
        self.series_entries.add_column("Series Number", lambda x: x.seriesNumber)
        self.series_entries.add_column("Image Count", lambda x: x.imageCount)
        self.series_entries_layout.addWidget(self.series_entries)

        self.series_view_control_container = QWidget(self.series_entries_container)
        self.series_view_control_layout = QHBoxLayout(self.series_view_control_container)
        self.series_view_control_layout.addWidget(self.back_to_beginning_button)
        self.series_view_control_layout.addWidget(self.back_to_studies_button)
        self.series_view_control_layout.addStretch()
        self.series_view_control_layout.addWidget(self.series_highlighted_label)
        self.series_view_control_layout.addWidget(self.open_button)
        self.series_entries_layout.addWidget(self.series_view_control_container)
        self.series_entries.get_selection.connect(self._on_series_table_double_clicked)
        self.series_entries.launch()
        self.back_to_studies_button.clicked.connect(lambda: self._on_back_to_studies_button_clicked())
        self.back_to_beginning_button.clicked.connect(lambda: self._on_back_to_search_button_clicked())
        self.interface_stack.addWidget(self.series_entries_container)

        self.study_entries_layout.setContentsMargins(0, 0, 0, 0)
        self.study_view_control_layout.setContentsMargins(0, 0, 0, 0)
        self.series_entries_layout.setContentsMargins(0, 0, 0, 0)
        self.series_view_control_layout.setContentsMargins(0, 0, 0, 0)

        self.refine_dataset_button.clicked.connect(lambda: self._on_drill_down_dataset_clicked())
        self.refine_study_button.clicked.connect(lambda: self._on_drill_down_clicked())
        self.open_button.clicked.connect(lambda: self._on_open_button_clicked())
        self.parent.setLayout(self.main_layout)
        self.tool_window.manage('side')

    def _on_search_button_pressed(self):
        if self.available_dbs.currentText() == "TCIA":
            # As with blastprotein, dicts cannot be used as table
            # entries because they are unhashable.
            # self._build_database_entries('tcia')?
            self.database_entries.data = [
                MainTableEntry(x['criteria'], x['count']) for x in fetch_nbia_collections_with_patients()
            ]

    def _on_main_table_double_clicked(self, items):
        # There'll only ever be one item from a double click
        entries = []
        for item in items:
            entries.extend([
                StudyTableEntry(
                    x['StudyInstanceUID']
                    , x['StudyDate']
                    , x['StudyDescription']
                    , x['PatientID']
                    , x['PatientName']
                    , x['PatientSex']
                    , x['SeriesCount']
                    , x['Collection']
                )
                for x in fetch_nbia_study(collection = item.collection)
            ])
        self.study_entries.data = entries
        self.interface_stack.setCurrentIndex(1)

    def _on_back_to_search_button_clicked(self):
        self.interface_stack.setCurrentIndex(0)

    def _on_back_to_studies_button_clicked(self):
        self.interface_stack.setCurrentIndex(1)

    def _on_study_table_double_clicked(self, items):
        # There'll only ever be one item from a double click
        entries = []
        for item in items:
            entries.extend([
                SeriesTableEntry(
                    x.get('SeriesInstanceUID', None)
                    , x.get('Modality', None)
                    , x.get('ProtocolName', None)
                    , x.get('SeriesDescription', None)
                    , x.get('BodyPartExamined', None)
                    , x.get('SeriesNumber', None)
                    , x.get('PatientID', None)
                    , x.get('ImageCount', None)
                )
                for x in fetch_nbia_series(studyUid = item.suid)
            ])
        self.series_entries.data = entries
        self.interface_stack.setCurrentIndex(2)

    def _on_drill_down_clicked(self):
        self._on_study_table_double_clicked(self.study_entries.selected)

    def _on_drill_down_dataset_clicked(self):
        self._on_main_table_double_clicked(self.database_entries.selected)

    def _on_open_button_clicked(self):
        self._on_series_table_double_clicked(self.series_entries.selected)

    def _on_series_table_double_clicked(self, items):
        for item in items:
            run(self.session, f"open {item.serUid} fromDatabase tcia format dicom")


class SeriesTableEntry:
    __slots__ = ["serUid", "modality", "protocolName", "seriesDescription", "bodyPart", "seriesNumber", "patientID",
                 "imageCount"]

    def __init__(self, serUid, modality, protocolName, seriesDescription, bodyPart, seriesNumber, patientID, imageCount):
        self.serUid = serUid
        self.modality = modality
        self.protocolName = protocolName
        self.seriesDescription = seriesDescription
        self.bodyPart = bodyPart
        self.seriesNumber = seriesNumber
        self.patientID = patientID
        self.imageCount = imageCount


class StudyTableEntry:
    __slots__ = ["suid", "date", "desc", "patientId", "patientName", "patientSex", "scount", "collection"]
    def __init__(self, suid, date, desc, patientId, patientName, patientSex, scount, collection):
        self.suid = suid
        self.date = date
        self.desc = desc
        self.patientId = patientId
        self.patientSex = patientSex
        self.patientName = patientName
        self.scount = scount
        self.collection = collection


class MainTableEntry:
    __slots__ = ["collection", "count"]

    def __init__(self, collection, count):
        self.collection = collection
        self.count = count
