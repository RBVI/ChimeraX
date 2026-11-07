"""Startup collector changes are bounded and preserve the caller's policy."""
import gc
from types import SimpleNamespace

import pytest

from chimerax.core import startup_gc


@pytest.fixture
def collector(monkeypatch):
    state = SimpleNamespace(enabled=True, frozen=0, freezes=0, unfreezes=0)

    def freeze():
        state.frozen = 100
        state.freezes += 1

    def unfreeze():
        state.frozen = 0
        state.unfreezes += 1

    monkeypatch.setattr(startup_gc, 'gc', SimpleNamespace(
        isenabled=lambda: state.enabled,
        enable=lambda: setattr(state, 'enabled', True),
        disable=lambda: setattr(state, 'enabled', False),
        get_freeze_count=lambda: state.frozen,
        freeze=freeze, unfreeze=unfreeze))
    return state


@pytest.fixture
def timers(monkeypatch):
    from Qt.QtCore import QTimer

    callbacks = []
    monkeypatch.setattr(QTimer, 'singleShot', lambda delay, cb: callbacks.append((delay, cb)))
    return callbacks


def test_collection_resumes_before_user_code_and_freeze_ends_once(collector, timers):
    with startup_gc.StartupGC(freeze_allowed=True) as policy:
        policy.pause()
        assert not collector.enabled
        policy.resume()
        assert collector.enabled
        assert collector.frozen == 100
        assert [delay for delay, callback in timers] == [1000]
        timers[0][1]()
        assert collector.frozen == 0
    assert collector.freezes == collector.unfreezes == 1


def test_failed_startup_restores_collector_without_freezing(collector):
    with pytest.raises(RuntimeError, match='startup failed'):
        with startup_gc.StartupGC(freeze_allowed=True) as policy:
            policy.pause()
            raise RuntimeError('startup failed')
    assert collector.enabled
    assert collector.freezes == collector.unfreezes == 0


def test_early_exit_unfreezes_without_waiting_for_event_loop(collector, timers):
    with startup_gc.StartupGC(freeze_allowed=True) as policy:
        policy.pause()
        policy.resume()
    assert collector.enabled
    assert collector.frozen == 0
    # A callback left in an embedding application's event queue is harmless.
    collector.frozen = 200
    timers[0][1]()
    assert collector.frozen == 200
    assert collector.unfreezes == 1


def test_existing_disabled_collector_is_left_disabled(collector, timers):
    collector.enabled = False
    with startup_gc.StartupGC(freeze_allowed=True) as policy:
        policy.pause()
        policy.resume()
    assert not collector.enabled
    assert collector.freezes == collector.unfreezes == 0
    assert not timers


def test_existing_frozen_heap_is_left_frozen(collector, timers):
    collector.frozen = 500
    with startup_gc.StartupGC(freeze_allowed=True) as policy:
        policy.pause()
        policy.resume()
    assert collector.enabled
    assert collector.frozen == 500
    assert collector.freezes == collector.unfreezes == 0
    assert not timers


def test_embedded_startup_does_not_freeze_the_host_heap(collector, timers):
    with startup_gc.StartupGC() as policy:
        policy.pause()
        assert not collector.enabled
        policy.resume()
        assert collector.enabled
    assert collector.freezes == collector.unfreezes == 0
    assert not timers


def test_user_can_disable_collection_after_startup(collector, timers):
    with startup_gc.StartupGC(freeze_allowed=True) as policy:
        policy.pause()
        policy.resume()
        collector.enabled = False
    assert not collector.enabled
    assert collector.frozen == 0


def test_timer_creation_failure_restores_heap(collector, monkeypatch):
    from Qt.QtCore import QTimer

    def fail(*args):
        raise RuntimeError('no timer')

    monkeypatch.setattr(QTimer, 'singleShot', fail)
    with startup_gc.StartupGC(freeze_allowed=True) as policy:
        policy.pause()
        policy.resume()
        assert collector.frozen == 0
    assert collector.enabled
    assert collector.frozen == 0


def test_init_restores_actual_collector_when_initialization_raises(monkeypatch):
    from chimerax.core import __main__ as main

    original = gc.isenabled()
    try:
        gc.enable()

        def fail(argv, event_loop, policy):
            policy.pause()
            assert not gc.isenabled()
            raise RuntimeError('startup failed')

        monkeypatch.setattr(main, '_init', fail)
        with pytest.raises(RuntimeError, match='startup failed'):
            main.init(['ChimeraX'])
        assert gc.isenabled()
    finally:
        (gc.enable if original else gc.disable)()
