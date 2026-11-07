"""Only normal standalone macOS GUI startup may defer session construction."""
from types import SimpleNamespace

import pytest

from chimerax.core import __main__ as main


@pytest.fixture
def options(monkeypatch):
    monkeypatch.setattr(main, 'sys', SimpleNamespace(platform='darwin'))
    monkeypatch.delenv('QT_QPA_PLATFORM', raising=False)
    return SimpleNamespace(gui=True, offscreen=False, safe_mode=False, uninstall=False, event_loop=True)


def test_normal_cocoa_startup_can_overlap(options):
    assert main._defer_gui_session_state(options, True)


@pytest.mark.parametrize('field,value', [('gui', False), ('offscreen', True),
    ('safe_mode', True), ('uninstall', True), ('event_loop', False)])
def test_special_launches_keep_synchronous_state(options, field, value):
    setattr(options, field, value)
    assert not main._defer_gui_session_state(options, True)


def test_embedding_keeps_synchronous_state(options):
    assert not main._defer_gui_session_state(options, False)


@pytest.mark.parametrize('platform', ['linux', 'win32'])
def test_other_platforms_are_unchanged(options, monkeypatch, platform):
    monkeypatch.setattr(main, 'sys', SimpleNamespace(platform=platform))
    assert not main._defer_gui_session_state(options, True)


@pytest.mark.parametrize('backend', ['offscreen', 'minimal', 'xcb', 'cocoa;offscreen'])
def test_nondefault_backends_keep_synchronous_state(options, monkeypatch, backend):
    monkeypatch.setenv('QT_QPA_PLATFORM', backend)
    assert not main._defer_gui_session_state(options, True)
