"""Opening the main window should not wait for an AppKit animation."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def cocoa(monkeypatch):
    import ctypes
    from chimerax.ui import gui
    calls = []
    def selector(name):
        return name
    objc = SimpleNamespace(sel_registerName=selector)
    monkeypatch.setattr(gui.sys, 'platform', 'darwin')
    monkeypatch.setattr(gui.QApplication, 'platformName', staticmethod(lambda: 'cocoa'))
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: objc)
    def signature(result_type, *argument_types):
        def bind(target):
            assert target == ('objc_msgSend', objc)
            if result_type is ctypes.c_void_p:
                def window(view, name):
                    calls.append(('window', view, name))
                    return 200
                return window
            def set_behavior(window, name, behavior):
                calls.append(('behavior', window, name, behavior))
            return set_behavior
        return bind
    monkeypatch.setattr(ctypes, 'CFUNCTYPE', signature)
    return gui, calls


def test_animation_is_disabled_on_the_native_window_before_show(cocoa):
    gui, calls = cocoa
    widget = SimpleNamespace(winId=lambda: 100)
    assert gui._disable_window_animation(widget)
    assert calls == [('window', 100, b'window'),
                     ('behavior', 200, b'setAnimationBehavior:', 2)]


@pytest.mark.parametrize('platform', ['win32', 'linux'])
def test_other_platforms_do_not_request_native_handles(cocoa, monkeypatch, platform):
    gui, calls = cocoa
    monkeypatch.setattr(gui.sys, 'platform', platform)
    assert not gui._disable_window_animation(object())
    assert calls == []


def test_missing_runtime_preserves_normal_window_show(cocoa, monkeypatch):
    import ctypes
    gui, calls = cocoa
    def unavailable(path):
        raise OSError('runtime unavailable')
    monkeypatch.setattr(ctypes, 'CDLL', unavailable)
    assert not gui._disable_window_animation(SimpleNamespace(winId=lambda: 100))
    assert calls == []


def test_offscreen_backend_never_treats_a_qt_id_as_an_nsview(cocoa, monkeypatch):
    gui, calls = cocoa
    monkeypatch.setattr(gui.QApplication, 'platformName', staticmethod(lambda: 'offscreen'))
    assert not gui._disable_window_animation(object())
    assert calls == []
