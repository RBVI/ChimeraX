"""The welcome banner is only needed while File History is empty."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def history_view(monkeypatch):
    from Qt.QtWidgets import QApplication, QWidget
    from chimerax.ui import file_history

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    history = SimpleNamespace(files=[object()], sections=[],
        find_file=lambda *key: None)
    monkeypatch.setattr(file_history.FileHistory, 'history', property(lambda self: history))
    monkeypatch.setattr(file_history.HistoryWindow, 'fill', lambda *args: None)
    session = SimpleNamespace(triggers=SimpleNamespace(add_handler=lambda *args: None))
    parent = QWidget()
    view = file_history.FileHistory(session, parent, show_banner=True, bg_color='white')
    yield view, history
    parent.close()
    parent.deleteLater()


def test_populated_history_does_not_create_welcome_banner(history_view):
    view, history = history_view
    assert view._empty_window is None
    assert view._stack.count() == 1
    assert view._stack.currentWidget() is view._history_window


def test_banner_created_on_empty_history_and_reused(history_view):
    from Qt.QtGui import QColor, QPalette

    view, history = history_view
    view.set_background_color('navy')
    history.files.clear()
    view.update()
    empty = view._empty_window
    assert empty is not None
    assert empty._title.text() == 'ChimeraX'
    assert empty.palette().color(QPalette.ColorRole.Window) == QColor('navy')
    assert view._stack.currentWidget() is empty
    history.files.append(object())
    view.update()
    assert view._stack.currentWidget() is view._history_window
    history.files.clear()
    view.update()
    assert view._stack.currentWidget() is empty
    assert view._stack.count() == 2


def test_empty_section_still_uses_history_view(history_view):
    view, history = history_view
    history.files.clear()
    history.sections.append('Work')
    view.update()
    assert view._empty_window is None
    assert view._stack.currentWidget() is view._history_window
