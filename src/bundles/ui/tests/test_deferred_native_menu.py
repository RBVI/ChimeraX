"""Attach the macOS native menu after painting, retaining other platforms' behavior."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('platform,disabled', [('darwin', False), ('darwin', True), ('linux', False)])
def test_menu_population_preserves_native_preference(monkeypatch, platform, disabled):
    from chimerax.ui import gui

    calls = []
    state = {'disabled': disabled}
    app = SimpleNamespace(testAttribute=lambda attr: state['disabled'])
    def set_attribute(attr, value):
        calls.append(('attribute', value))
        state['disabled'] = value
    app.setAttribute = set_attribute
    monkeypatch.setattr(gui, 'QApplication', app)
    monkeypatch.setattr(gui, 'sys', SimpleNamespace(platform=platform))
    menu = SimpleNamespace(hide=lambda: calls.append('hide'))
    def menu_bar():
        calls.append(('create', state['disabled']))
        return menu
    window = SimpleNamespace(menuBar=menu_bar,
        _populate_menu_contents=lambda ses, mb: calls.append(('populate', mb)))
    gui.MainWindow._populate_menus(window, object())
    assert state['disabled'] is disabled
    if platform == 'darwin' and not disabled:
        assert calls == [('attribute', True), ('create', True), ('attribute', False),
                         'hide', ('populate', menu)]
        assert window._native_menu_to_attach is menu
    else:
        assert calls == [('create', disabled), ('populate', menu)]
        assert not hasattr(window, '_native_menu_to_attach')


def test_first_paint_attaches_once_after_event_returns(monkeypatch):
    from Qt.QtCore import QTimer
    from Qt import QtWidgets
    from chimerax.ui import gui

    calls, callbacks = [], []
    monkeypatch.setattr(gui, 'QMainWindow', SimpleNamespace(
        paintEvent=lambda self, event: calls.append('paint')))
    monkeypatch.setattr(QTimer, 'singleShot', lambda delay, cb: callbacks.append(cb))
    child = SimpleNamespace(setParent=lambda parent: calls.append(('parent', parent)))
    action = SimpleNamespace(menu=lambda: child)
    menu = SimpleNamespace(actions=lambda: [action])
    native = SimpleNamespace(addAction=lambda action: calls.append(('action', action)))
    monkeypatch.setattr(QtWidgets, 'QMenuBar', lambda parent: native)
    window = SimpleNamespace(_native_menu_to_attach=menu, _native_menus_ready=True,
        setMenuBar=lambda bar: calls.append(('bar', bar)))
    gui.MainWindow.paintEvent(window, object())
    gui.MainWindow.paintEvent(window, object())
    assert calls == ['paint', 'paint']
    assert window._native_menu_to_attach is None
    assert len(callbacks) == 1
    callbacks[0]()
    assert calls == ['paint', 'paint', ('parent', native), ('action', action), ('bar', native)]


def test_partial_paint_during_tool_construction_does_not_attach_menus(monkeypatch):
    from chimerax.ui import gui

    calls = []
    monkeypatch.setattr(gui, 'QMainWindow', SimpleNamespace(
        paintEvent=lambda self, event: calls.append('paint')))
    menu = object()
    window = SimpleNamespace(_native_menu_to_attach=menu)
    gui.MainWindow.paintEvent(window, object())
    assert calls == ['paint']
    assert window._native_menu_to_attach is menu


def test_creation_failure_restores_application_attribute(monkeypatch):
    from chimerax.ui import gui

    state = {'disabled': False}
    app = SimpleNamespace(testAttribute=lambda attr: state['disabled'],
        setAttribute=lambda attr, value: state.update(disabled=value))
    monkeypatch.setattr(gui, 'QApplication', app)
    monkeypatch.setattr(gui, 'sys', SimpleNamespace(platform='darwin'))
    def fail():
        raise RuntimeError('menu creation failed')
    with pytest.raises(RuntimeError, match='menu creation failed'):
        gui.MainWindow._populate_menus(SimpleNamespace(menuBar=fail), object())
    assert not state['disabled']
