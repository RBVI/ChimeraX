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
from enum import Enum

from Qt.QtCore import QThread, QObject, Signal, Slot, Qt
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QHeaderView
    , QWidget, QLabel, QAbstractItemView
    , QPushButton, QAction, QComboBox
    , QStackedWidget, QSizePolicy
)

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run
from chimerax.help_viewer import show_url

from ..databases import TCIADatabase
from .widgets import DICOMTable

class Action(Enum):
    LOAD_COLLECTIONS = 1
    LOAD_STUDIES = 2
    LOAD_SERIES = 3

class DICOMDatabases(ToolInstance):

    help = "help:user/tools/downloaddicom.html"

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
        # TODO: Remove when there's more than one DB, add 'Choose a Database' item to spinbox
        self.database_label = QLabel("Database:", self.database_entries_container)
        self.control_container = QWidget(self.database_entries_container)
        self.control_layout = QHBoxLayout(self.control_container)

        self.control_container.setLayout(self.control_layout)

        # TODO: Remove when there's more than one DB, add 'Choose a Database' item to spinbox
        self.control_layout.addWidget(self.database_label)
        self.control_layout.addWidget(self.available_dbs)
        self.control_layout.addStretch()
        self.dataset_highlighted_label = QLabel("For chosen entries:")
        self.load_webpage_button = QPushButton("Load Webpage")
        self.refine_dataset_button = QPushButton("Drill Down to Studies")
        self.control_layout.addWidget(self.dataset_highlighted_label)
        self.control_layout.addWidget(self.load_webpage_button)
        self.control_layout.addWidget(self.refine_dataset_button)
        self.database_entries_control_widget.setVisible(False)
        self.database_entries = DICOMTable(self.database_entries_control_widget, None, self.database_entries_container)
        self.database_entries_layout.addWidget(self.database_entries)
        self.database_entries_layout.addWidget(self.database_entries_control_widget)
        self.database_entries_layout.addWidget(self.control_container)
        self.database_entries_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setContentsMargins(4,4,4,4)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.database_entries_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.database_entries_layout.setSpacing(0)
        self.control_layout.setSpacing(0)
        self.main_layout.setSpacing(0)

        self.database_entries.add_column("Dataset", lambda x: x.collection)
        self.database_entries.add_column("Number of Patients", lambda x: x.count)
        self.database_entries.add_column("Body Parts", lambda x: x.body_parts)
        self.database_entries.add_column("Species", lambda x: x.species)
        self.database_entries.add_column("Modalities", lambda x: x.modalities)

        self.interface_stack.addWidget(self.database_entries_container)

        self.combo_box_model = self.available_dbs.model()
        # TODO: Enable when there's more than one database
        # self.available_dbs.addItem("Choose a Database")
        self.available_dbs.addItem("TCIA")
        self.database_entries.launch()
        self.database_entries.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.interface_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.database_entries.get_selection.connect(self._on_main_table_double_clicked)

        self.study_entries_container = QWidget(self.interface_stack)
        self.study_entries_layout = QVBoxLayout(self.study_entries_container)
        self.back_to_search_button = QPushButton("Back to Collections")
        self.study_highlighted_label = QLabel("For chosen entries:")
        self.refine_study_button = QPushButton("Drill Down to Series")

        self.study_entries_control_widget = QWidget(self.study_entries_container)
        self.study_entries_control_widget.setVisible(False)
        self.study_entries = DICOMTable(self.study_entries_control_widget, None, self.study_entries_container)
        self.study_entries.add_column("Collection", lambda x: x.collection)
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
        self.series_highlighted_label = QLabel("For chosen entries:")
        self.open_button = QPushButton("Download and Open")

        self.series_entries_control_widget = QWidget(self.series_entries_container)
        self.series_entries_control_widget.setVisible(False)
        self.series_entries = DICOMTable(self.series_entries_control_widget, None, self.series_entries_container)
        self.series_entries.add_column("Study Instance UID", lambda x: x.studyUid)
        self.series_entries.add_column("Series Instance UID", lambda x: x.seriesUid)
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

        self.load_webpage_button.clicked.connect(lambda: self._on_open_tcia_webpage())
        self.refine_dataset_button.clicked.connect(lambda: self._on_drill_down_dataset_clicked())
        self.refine_study_button.clicked.connect(lambda: self._on_drill_down_clicked())
        self.open_button.clicked.connect(lambda: self._on_open_button_clicked())
        self.parent.setLayout(self.main_layout)
        self.tool_window.fill_context_menu = self._fill_context_menu
        self.tool_window.manage('side')

        # TODO: When there's more than one database available remove this call
        self._on_database_changed()

    def _fill_context_menu(self, menu, x, y):
        table = self._hovered_table(x, y)
        if table:
            load_database_action = QAction("Load Webpage for Chosen Entries", menu)
            load_database_action.triggered.connect(lambda: self._on_open_tcia_webpage(self.database_entries.selected))
            menu.addAction(load_database_action)

    def _hovered_table(self, x, y):
        widget = self.tool_window.ui_area.childAt(x, y)
        if not widget:
            return None
        if widget == self.database_entries:
            return widget
        if widget.parent() == self.database_entries:
            return widget.parent()
        return None

    def _on_open_tcia_webpage(self, selections = None):
        if not selections:
            selections = self.database_entries.selected
        for selection in selections:
            if selection.url is not None:
                show_url(self.session, selection.url, new_tab=True)

    def _allocate_thread_and_worker(self, action: Action):
        self.thread = QThread()
        self.worker = DatabaseWorker(self.session, self.available_dbs.currentText(), action)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.worker.deleteLater)

    def _on_database_changed(self):
        self._allocate_thread_and_worker(Action.LOAD_COLLECTIONS)
        self.worker.collections_ready.connect(self._on_collection_entries_returned_from_worker)
        self.thread.start()

    def _on_collection_entries_returned_from_worker(self, entries):
        self.database_entries.data = [
            MainTableEntry(
                x['name']
                , x['patients']
                , x['body_parts']
                , x['species']
                , x['modalities']
                , x['url']
            ) for x in entries
        ]

    def _on_main_table_double_clicked(self, items):
        self._allocate_thread_and_worker(Action.LOAD_STUDIES)
        self.worker.studies_ready.connect(self._on_studies_returned_from_worker)
        self.worker.requested_studies = items
        self.thread.start()

    def _on_studies_returned_from_worker(self, entries):
        self.study_entries.data = [
            StudyTableEntry(
                  x.get('StudyInstanceUID', None)
                , x.get('StudyDate', None)
                , x.get('StudyDescription', None)
                , x.get('PatientID', None)
                , x.get('PatientName', None)
                , x.get('PatientSex', None)
                , x.get('SeriesCount', None)
                , x.get('Collection', None)
            )
            for x in entries
        ]
        self.study_entries.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.interface_stack.setCurrentIndex(1)

    def _on_back_to_search_button_clicked(self):
        self.interface_stack.setCurrentIndex(0)

    def _on_back_to_studies_button_clicked(self):
        self.interface_stack.setCurrentIndex(1)

    def _on_study_table_double_clicked(self, items):
        self._allocate_thread_and_worker(Action.LOAD_SERIES)
        self.worker.series_ready.connect(self._on_series_returned_from_worker)
        self.worker.requested_series = items
        self.thread.start()

    def _on_series_returned_from_worker(self, entries):
        self.series_entries.data = [
            SeriesTableEntry(
                x.get('StudyInstanceUID', None)
                , x.get('SeriesInstanceUID', None)
                , x.get('Modality', None)
                , x.get('ProtocolName', None)
                , x.get('SeriesDescription', None)
                , x.get('BodyPartExamined', None)
                , x.get('SeriesNumber', None)
                , x.get('PatientID', None)
                , x.get('ImageCount', None)
            )
            for x in entries
        ]
        self.series_entries.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.interface_stack.setCurrentIndex(2)

    def _on_drill_down_clicked(self):
        self._on_study_table_double_clicked(self.study_entries.selected)

    def _on_drill_down_dataset_clicked(self):
        self._on_main_table_double_clicked(self.database_entries.selected)

    def _on_open_button_clicked(self):
        self._on_series_table_double_clicked(self.series_entries.selected)

    def _on_series_table_double_clicked(self, items):
        for item in items:
            run(self.session, f"open {item.seriesUid} fromDatabase tcia format dicom")


class SeriesTableEntry:
    __slots__ = ["studyUid", "seriesUid", "modality", "protocolName", "seriesDescription", "bodyPart", "seriesNumber", "patientID",
                 "imageCount"]

    def __init__(self, studyUid, serUid, modality, protocolName, seriesDescription, bodyPart, seriesNumber, patientID, imageCount):
        self.studyUid = studyUid
        self.seriesUid = serUid
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
    __slots__ = ["collection", "count", "body_parts", "species", "modalities", "url"]

    def __init__(self, collection, count, body_parts, species, modalities, url):
        self.collection = collection
        self.count = count
        self.body_parts = ", ".join(body_parts)
        self.modalities = ", ".join(modalities)
        self.species = ", ".join(species)
        self.url = url

class DatabaseWorker(QObject):
    collections_ready = Signal(list)
    studies_ready = Signal(list)
    series_ready = Signal(list)
    finished = Signal()

    def __init__(self, session, database: str, action: Action):
        super().__init__()
        self.session = session
        self.database = database
        self.action = action
        self.requested_studies = None
        self.requested_series = None

    @Slot()
    def run(self):
        if self.action == Action.LOAD_COLLECTIONS:
            self._fetch_collections()
        elif self.action == Action.LOAD_STUDIES:
            self._fetch_studies()
        elif self.action == Action.LOAD_SERIES:
            self._fetch_series()
        self.finished.emit()

    def _fetch_collections(self):
        entries = None
        self.session.ui.thread_safe(self.session.logger.status, f"Loading collections from {self.database}")
        if self.database == "TCIA":
            entries = TCIADatabase.get_collections(self.session)
        self.collections_ready.emit(entries)

    def _fetch_studies(self):
        entries = []
        self.session.ui.thread_safe(self.session.logger.status, "Loading requested studies...")
        if self.database == "TCIA":
            for study in self.requested_studies:
                entries.extend(TCIADatabase.get_study(collection=study.collection))
        self.studies_ready.emit(entries)

    def _fetch_series(self):
        entries = []
        self.session.ui.thread_safe(self.session.logger.status, "Loading requested series...")
        if self.database == "TCIA":
            for series in self.requested_series:
                entries.extend(TCIADatabase.get_series(studyUid=series.suid))
        self.series_ready.emit(entries)
