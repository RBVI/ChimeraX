# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import string
import datetime
from typing import Dict, Optional, Union

from Qt.QtCore import QThread, QObject, Signal, Slot, Qt
from Qt.QtWidgets import (
    QPushButton, QSizePolicy
    , QVBoxLayout, QHBoxLayout, QComboBox
    , QWidget, QSpinBox, QAbstractSpinBox
    , QStackedWidget, QPlainTextEdit
    , QLineEdit, QPushButton, QMenu
)

from chimerax.core.commands import run
from chimerax.core.settings import Settings
from chimerax.core.tasks import TaskState
from chimerax.core.tools import ToolInstance

from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import ItemTable


class TaskManagerTable(ItemTable):
    def __init__(self, control_widget: Union[QMenu, QWidget], default_cols, settings: 'TaskManagerTableSettings', parent = Optional[QWidget]):
        super().__init__(
            column_control_info=(
                control_widget
                , settings
                , default_cols
                , False        # fallback default for column display
                , None         # display callback
                , None         # number of checkbox columns
                , True         # Whether to show global buttons
            )
            , parent=parent)


class TaskManagerTableSettings(Settings):
    EXPLICIT_SAVE = {TaskManagerTable.DEFAULT_SETTINGS_ATTR: {}}


class TaskManager(ToolInstance):
    SESSION_ENDURING = True
    SESSION_SAVE = False
    help = "help:user/tools/taskmanager.html"

    def __init__(self, session, name = "Task Manager"):
        super().__init__(session, name)
        self._build_ui()

    def _build_ui(self):
        self.tool_window = MainToolWindow(self)
        self.parent = self.tool_window.ui_area
        self.clear_finished_button = QPushButton("Clear Finished")
        self.kill_task_button = QPushButton("Kill Task(s)")
        self.close_button = QPushButton("Close")
        self.help_button = QPushButton("Help")

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.button_container = QWidget()
        self.button_container_layout = QHBoxLayout()
        self.button_container.setLayout(self.button_container_layout)
        self.button_container_layout.setContentsMargins(0, 0, 0, 0)
        self.button_container_layout.setSpacing(0)
        self.button_container_layout.addStretch()
        self.button_container_layout.addWidget(self.kill_task_button)
        self.button_container_layout.addWidget(self.clear_finished_button)
        self.button_container_layout.addWidget(self.close_button)
        self.button_container_layout.addWidget(self.help_button)

        self.kill_task_button.clicked.connect(lambda: self.kill_task(self.table.selected))
        self.help_button.clicked.connect(self.show_help)
        self.close_button.clicked.connect(self.delete)
        self.clear_finished_button.clicked.connect(self._clear_finished)

        self.parent.setLayout(self.main_layout)
        self.parent.layout().addWidget(self.button_container)
        # ID, Task, Status, Started, Run Time, State
        self.control_widget = QWidget(self.parent)
        self.table = TaskManagerTable(self.control_widget, {"ID": True, "Task": True}, None, self.parent)
        self.table.data = [task for task in self.session.tasks.values()]
        self.table.add_column("ID (Local)", data_fetch=lambda x: x.id)
        self.table.add_column("ID (Webservices)", data_fetch=lambda x: getattr(x, "job_id", None))
        self.table.add_column("Task", data_fetch=lambda x: str(x))
        self.table.add_column("Start Time", data_fetch=lambda x: x.start_time.strftime("%H:%M:%S%p"))
        self.table.add_column("Runtime", data_fetch=lambda x: self._get_runtime(x.runtime))
        self.table.add_column("Status", data_fetch=lambda x: x.state)
        self.table.launch(suppress_resize=True)
        self.control_widget.setVisible(True)
        self.parent.layout().addWidget(self.table)
        self.parent.layout().addWidget(self.control_widget)
        self.parent.layout().addWidget(self.button_container)
        self.thread = QThread()
        self.worker = TaskManagerWorker(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.update_table.connect(self._refresh_table)
        self.thread.start()
        self.tool_window.manage('side')

    def delete(self):
        try:
            self.worker.blockSignals(True)
        except RuntimeError:
            pass  # The underlying C++ object has already been deleted by DeleteLater
        try:
            if self.thread is not None and self.thread.isRunning():
                self.thread.exit()
        except RuntimeError:
            pass # The underlying C++ object has already been deleted by DeleteLater
        super().delete()

    def _refresh_table(self, *args):
        self.table.data = [task for task in self.session.tasks.values()]
        self.table.viewport().update()

    def _get_runtime(self, timedelta: datetime.timedelta) -> str:
        hours = timedelta.seconds // 3600
        minutes = (timedelta.seconds // 60) % 60
        seconds = timedelta.seconds - (60 * minutes)
        return "%sh %sm %ss" % (hours, minutes, seconds)

    def _clear_finished(self):
        tasks_to_delete = []
        for id, task in self.session.tasks.items():
            if task.state == TaskState.FINISHED:
                # Can't change the size of a dict during iteration
                tasks_to_delete.append(id)
        for id in tasks_to_delete:
            del self.session.tasks[id]
        self._refresh_table()

    def _format_column_title(self, s):
        return string.capwords(s.replace('_', ' '))

    def show_help(self):
        run(self.session, "help 'Task Manager'")

    def kill_task(self, tasks):
        for task in tasks:
            run(self.session, "taskman kill %s" % task.id)


class TaskManagerWorker(QObject):
    update_table = Signal()

    def __init__(self, tool):
        super().__init__()
        self.tool = tool

    @Slot()
    def run(self) -> None:
        while True:
            self.update_table.emit()
            QThread.sleep(1)
