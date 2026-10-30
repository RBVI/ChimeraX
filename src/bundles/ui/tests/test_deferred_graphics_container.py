"""First-use native window creation must preserve viewport geometry and lifetime."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def qt_app():
    from Qt.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    return app


def test_hidden_graphics_does_not_create_native_container(qt_app, monkeypatch):
    from Qt.QtGui import QWindow
    from Qt.QtWidgets import QWidget
    from chimerax.ui.graphics import _DeferredWindowContainer

    def unexpected(*args):
        pytest.fail('Hidden graphics created a native container')

    monkeypatch.setattr(QWidget, 'createWindowContainer', unexpected)
    window = QWindow()
    panel = _DeferredWindowContainer(window, None)
    try:
        panel.resize(640, 480)
        qt_app.processEvents()
        assert panel._container is None
        assert not window.isVisible()
    finally:
        panel.deleteLater()


def test_first_show_sets_viewport_before_embedding_and_reuses_container(qt_app, monkeypatch):
    from Qt.QtGui import QWindow
    from Qt.QtWidgets import QWidget
    from chimerax.ui.graphics import _DeferredWindowContainer

    sizes = []
    embedded = []
    window = QWindow()
    window.view = SimpleNamespace(resize=lambda w, h: sizes.append((w, h)))

    def embed(target, parent):
        # Camera fitting may run before Qt delivers the native resize event.
        assert target is window
        assert sizes[-1] == (parent.width(), parent.height()) == (640, 480)
        child = QWidget(parent)
        embedded.append(child)
        return child

    monkeypatch.setattr(QWidget, 'createWindowContainer', embed)
    panel = _DeferredWindowContainer(window, None)
    try:
        panel.resize(640, 480)
        panel.show()
        qt_app.processEvents()
        assert len(embedded) == 1
        child = embedded[0]
        assert panel.focusProxy() is child
        assert child.acceptDrops()
        assert child.size() == panel.size()
        panel.hide()
        panel.resize(800, 600)
        panel.show()
        assert sizes[-1] == (800, 600)
        qt_app.processEvents()
        assert embedded == [child]
        assert child.size() == panel.size()
    finally:
        panel.close()
        panel.deleteLater()


def test_deleting_unshown_panel_releases_unembedded_window(qt_app):
    from Qt import qt_object_is_deleted
    from Qt.QtCore import QCoreApplication, QEvent
    from Qt.QtGui import QWindow
    from chimerax.ui.graphics import _DeferredWindowContainer

    window = QWindow()
    panel = _DeferredWindowContainer(window, None)
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert qt_object_is_deleted(window)
