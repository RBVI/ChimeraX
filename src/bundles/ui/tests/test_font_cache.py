"""Font warmup keeps its native value alive and finishes before UI teardown."""
from types import SimpleNamespace
import weakref

import pytest


@pytest.fixture
def native_font(monkeypatch):
    import ctypes
    import sys
    import Qt
    from Qt import QtGui
    shiboken6 = pytest.importorskip('shiboken6')
    from chimerax.ui import font

    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(Qt, 'using_pyside6', True)
    created = []
    class Metrics:
        def __init__(self, value):
            self.font = value
            created.append(weakref.ref(self))
    monkeypatch.setattr(QtGui, 'QFontMetrics', Metrics)
    monkeypatch.setattr(shiboken6, 'getCppPointer', lambda value: (1234,))
    calls = []
    class Height:
        def __call__(self, pointer):
            assert created[0]() is not None
            calls.append(pointer)
            return 17
    height = Height()
    monkeypatch.setattr(ctypes, 'CDLL', lambda path:
        SimpleNamespace(_ZNK12QFontMetrics6heightEv=height))
    return font, SimpleNamespace(font=lambda: 'default font'), calls


def test_native_value_lives_until_warmup_finishes(native_font):
    font, ui, calls = native_font
    worker = font._start_font_cache(ui)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert calls == [1234]


def test_unavailable_framework_keeps_synchronous_fallback(native_font, monkeypatch):
    import ctypes
    font, ui, calls = native_font
    def missing(path):
        raise OSError('framework not available')
    monkeypatch.setattr(ctypes, 'CDLL', missing)
    assert font._start_font_cache(ui) is None
    assert calls == []


def test_unavailable_method_keeps_synchronous_fallback(native_font, monkeypatch):
    import ctypes
    font, ui, calls = native_font
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: object())
    assert font._start_font_cache(ui) is None
    assert calls == []


def test_windows_does_not_start_native_work(native_font, monkeypatch):
    import sys
    font, ui, calls = native_font
    monkeypatch.setattr(sys, 'platform', 'win32')
    assert font._start_font_cache(ui) is None
    assert calls == []


def test_linux_warms_fonts_through_libqt6gui(native_font, monkeypatch):
    import ctypes
    import sys
    font, ui, calls = native_font
    monkeypatch.setattr(sys, 'platform', 'linux')
    opened = []
    cdll = ctypes.CDLL
    def record(path):
        opened.append(path)
        return cdll(path)
    monkeypatch.setattr(ctypes, 'CDLL', record)
    worker = font._start_font_cache(ui)
    worker.join(timeout=5)
    assert opened == ['libQt6Gui.so.6']
    assert calls == [1234]


def test_ui_waits_once_for_font_work():
    from chimerax.ui.gui import UI
    calls = []
    ui = SimpleNamespace(_font_cache_thread=SimpleNamespace(
        join=lambda: calls.append('joined')))
    UI._wait_for_font_cache(ui)
    assert calls == ['joined']
    assert ui._font_cache_thread is None
    UI._wait_for_font_cache(ui)
    assert calls == ['joined']


def test_other_binding_keeps_synchronous_fallback(native_font, monkeypatch):
    import Qt
    font, ui, calls = native_font
    monkeypatch.setattr(Qt, 'using_pyside6', False)
    assert font._start_font_cache(ui) is None
    assert calls == []


def test_thread_start_failure_keeps_synchronous_fallback(native_font, monkeypatch):
    import threading
    font, ui, calls = native_font
    class UnavailableThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            raise RuntimeError('cannot create thread')
    monkeypatch.setattr(threading, 'Thread', UnavailableThread)
    assert font._start_font_cache(ui) is None
    assert calls == []


def test_invalid_native_pointer_keeps_synchronous_fallback(native_font, monkeypatch):
    import shiboken6
    font, ui, calls = native_font
    def unavailable(value):
        raise TypeError('no native value')
    monkeypatch.setattr(shiboken6, 'getCppPointer', unavailable)
    assert font._start_font_cache(ui) is None
    assert calls == []


def test_native_failure_does_not_escape_worker(native_font, monkeypatch):
    import ctypes
    import threading
    font, ui, calls = native_font
    class FailingHeight:
        def __call__(self, pointer):
            calls.append(pointer)
            raise RuntimeError('native call unavailable')
    monkeypatch.setattr(ctypes, 'CDLL', lambda path:
        SimpleNamespace(_ZNK12QFontMetrics6heightEv=FailingHeight()))
    errors = []
    monkeypatch.setattr(threading, 'excepthook', errors.append)
    worker = font._start_font_cache(ui)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert calls == [1234]
    assert errors == []


def test_build_waits_before_constructing_widgets(monkeypatch):
    from chimerax.ui import gui
    calls = []
    class StopBuild(Exception):
        pass
    def main_window(ui, session):
        calls.append('window')
        raise StopBuild
    monkeypatch.setattr(gui, 'MainWindow', main_window)
    ui = SimpleNamespace(session=object(),
        _wait_for_font_cache=lambda: calls.append('joined'))
    with pytest.raises(StopBuild):
        gui.UI.build(ui)
    assert calls == ['joined', 'window']
