"""Native menus use the original bar and retain the application's preference."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('platform,disabled', [('darwin', False), ('darwin', True), ('linux', False)])
def test_menu_population_preserves_native_preference(monkeypatch, platform, disabled):
    from chimerax.ui import gui

    calls = []
    state = {'disabled': disabled}

    def set_attribute(attr, value):
        calls.append(('attribute', value))
        state['disabled'] = value

    monkeypatch.setattr(gui, 'QApplication', SimpleNamespace(
        setAttribute=set_attribute, testAttribute=lambda attr: state['disabled']))
    monkeypatch.setattr(gui, 'sys', SimpleNamespace(platform=platform))
    menu = object()

    def menu_bar():
        calls.append(('create', state['disabled']))
        return menu

    window = SimpleNamespace(menuBar=menu_bar,
        _populate_menu_contents=lambda ses, mb: calls.append(('populate', mb)))
    gui.MainWindow._populate_menus(window, object())
    assert state['disabled'] is disabled
    assert calls == [('create', disabled), ('populate', menu)]
    assert not hasattr(window, '_native_menu_to_attach')


def test_menu_creation_failure_does_not_change_application_preference(monkeypatch):
    from chimerax.ui import gui

    def unexpected(*args):
        pytest.fail('Menu population changed application attributes')

    monkeypatch.setattr(gui, 'QApplication', SimpleNamespace(setAttribute=unexpected))

    def fail():
        raise RuntimeError('menu creation failed')

    with pytest.raises(RuntimeError, match='menu creation failed'):
        gui.MainWindow._populate_menus(SimpleNamespace(menuBar=fail,
            _populate_menu_contents=unexpected), object())
