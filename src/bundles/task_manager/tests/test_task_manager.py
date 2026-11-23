import datetime
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


class _DummyTools:
    def __init__(self):
        self.items = []

    def add(self, tools):
        self.items.extend(tools)

    def remove(self, tools):
        for tool in tools:
            if tool in self.items:
                self.items.remove(tool)


class _DummySession:
    def __init__(self, tasks=None):
        self.tasks = dict(tasks or {})
        self.tools = _DummyTools()
        self.ui = SimpleNamespace(is_gui=False)


def _make_session(tasks=None):
    return _DummySession(tasks)


def _make_fake_tool_window():
    from Qt.QtWidgets import QWidget

    class _FakeMainToolWindow:
        def __init__(self, tool_instance):
            self.tool_instance = tool_instance
            self.ui_area = QWidget()
            self.managed = []

        def manage(self, placement):
            self.managed.append(placement)

    return _FakeMainToolWindow


def _ensure_qapp():
    from Qt.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication([])
    return app


def test_task_manager_uses_qtimer_and_stops_on_delete(monkeypatch):
    _ensure_qapp()
    from Qt.QtCore import QTimer
    from chimerax.taskmanager import tool as tm_tool

    monkeypatch.setattr(tm_tool, "MainToolWindow", _make_fake_tool_window())

    session = _make_session()
    tm = tm_tool.TaskManager(session)

    assert not hasattr(tm, "thread")
    assert isinstance(tm._refresh_timer, QTimer)
    assert tm._refresh_timer.parent() is tm.parent
    assert tm._refresh_timer.isActive()
    assert tm._refresh_timer.interval() == 1000

    class _DummyTask:
        id = 1
        job_id = "job-1"
        start_time = datetime.datetime(2026, 1, 1, 12, 0, 0)
        runtime = datetime.timedelta(seconds=5)
        state = "running"

    task = _DummyTask()
    session.tasks[1] = task
    tm._refresh_timer.timeout.emit()
    assert tm.table.data == [task]

    tm.delete()
    assert not tm._refresh_timer.isActive()

    # Calling delete() when the underlying Qt timer is already deleted must not raise.
    tm._refresh_timer.deleteLater()
    _ensure_qapp().processEvents()
    tm.delete()


@pytest.mark.parametrize("close_before_exit", [False, True])
def test_task_manager_clean_process_exit(close_before_exit):
    # Regression test for Trac #21383: opening Task Manager previously spawned
    # a QThread running an infinite sleep loop that caused Qt to abort with
    # "QThread: Destroyed while thread '' is still running" during PySide6
    # atexit cleanup.
    script = f"""
from types import SimpleNamespace
from Qt.QtWidgets import QApplication, QWidget
from chimerax.taskmanager import tool as tm_tool

app = QApplication([])

class _Tools:
    def add(self, tools): pass
    def remove(self, tools): pass

class _Window:
    def __init__(self, tool_instance):
        self.ui_area = QWidget()
    def manage(self, placement):
        pass

class _Session:
    def __init__(self):
        self.tasks = {{}}
        self.tools = _Tools()
        self.ui = SimpleNamespace(is_gui=False)

tm_tool.MainToolWindow = _Window
session = _Session()
tm = tm_tool.TaskManager(session)
app.processEvents()
if {close_before_exit!r}:
    tm.delete()
    app.processEvents()
"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
