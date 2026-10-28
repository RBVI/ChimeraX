"""The OpenGL driver is initialized in the background on Linux."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def warmup(monkeypatch):
    import ctypes
    import sys
    from chimerax.graphics import opengl
    from Qt.QtCore import QCoreApplication
    shiboken6 = pytest.importorskip('shiboken6')
    monkeypatch.setattr(opengl, '_driver_warmup', None)
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(QCoreApplication, 'testAttribute', staticmethod(lambda attr: True))
    monkeypatch.setattr(shiboken6, 'getCppPointer', lambda obj: (99,))
    calls = []

    class Function:
        def __init__(self, name, result=None):
            self.name, self.result = name, result
        def __call__(self, *args):
            calls.append((self.name,) + args)
            return self.result
    libs = {
        'libQt6Gui.so.6': SimpleNamespace(_ZN14QOpenGLContext18globalShareContextEv=Function('share', 1234)),
        'libQt6Core.so.6': SimpleNamespace(_ZN7QObject12moveToThreadEP7QThread=Function('move')),
    }
    monkeypatch.setattr(ctypes, 'CDLL', lambda name: libs[name])
    app = SimpleNamespace(thread=lambda: 'main thread')
    return opengl, app, calls


def test_share_context_is_created_and_moved_to_the_main_thread(warmup):
    opengl, app, calls = warmup
    assert opengl.start_driver_warmup(app)
    opengl._wait_for_driver_warmup()
    assert calls == [('share',), ('move', 1234, 99)]
    # Waiting again, or starting again, does nothing.
    opengl._wait_for_driver_warmup()
    assert not opengl.start_driver_warmup(app)
    assert len(calls) == 2


def test_waiting_without_warmup_does_nothing(warmup):
    opengl, app, calls = warmup
    opengl._wait_for_driver_warmup()
    assert calls == []


@pytest.mark.parametrize('platform', ['darwin', 'win32'])
def test_other_platforms_do_not_warm_up(warmup, monkeypatch, platform):
    import sys
    opengl, app, calls = warmup
    monkeypatch.setattr(sys, 'platform', platform)
    assert not opengl.start_driver_warmup(app)


def test_no_warmup_without_shared_contexts(warmup, monkeypatch):
    from Qt.QtCore import QCoreApplication
    opengl, app, calls = warmup
    monkeypatch.setattr(QCoreApplication, 'testAttribute', staticmethod(lambda attr: False))
    assert not opengl.start_driver_warmup(app)


def test_missing_qt_symbols_fall_back(warmup, monkeypatch):
    import ctypes
    opengl, app, calls = warmup
    monkeypatch.setattr(ctypes, 'CDLL', lambda name: SimpleNamespace())
    assert not opengl.start_driver_warmup(app)
    opengl._wait_for_driver_warmup()
    assert calls == []
