"""The error dialog is created once, retaining bug reporting and message queues."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def log():
    from Qt.QtWidgets import QApplication, QWidget
    from chimerax.log.tool import Log

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    parent = QWidget()
    instance = Log.__new__(Log)
    instance._error_dialog = None
    instance.tool_window = SimpleNamespace(ui_area=parent)
    instance.session = SimpleNamespace(ui=app)
    yield instance
    if instance._error_dialog is not None:
        instance._error_dialog.close()
    parent.close()
    parent.deleteLater()


def test_first_access_creates_owned_dialog_and_report_button_once(log):
    from Qt.QtCore import QSize
    from Qt.QtWidgets import QPushButton

    assert log._error_dialog is None
    dialog = log.error_dialog
    assert dialog is log.error_dialog is log._error_dialog
    assert dialog.parent() is log.tool_window.ui_area
    assert dialog.sizeHint() == QSize(600, 300)
    assert dialog.report_bug_button.text() == 'Report Bug'
    assert sum(button.text() == 'Report Bug' for button in dialog.findChildren(QPushButton)) == 1


def test_first_bug_and_later_error_reuse_dialog_with_correct_button_visibility(log):
    log.show_error_message('First bug', bug=True)
    dialog = log.error_dialog
    assert not dialog.report_bug_button.isHidden()
    log.show_error_message('Later ordinary error')
    assert log.error_dialog is dialog
    assert dialog.report_bug_button.isHidden()


def test_first_ordinary_error_also_initializes_the_report_button(log):
    log.show_error_message('First ordinary error')
    assert log._error_dialog is not None
    assert log.error_dialog.report_bug_button.isHidden()
