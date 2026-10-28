"""Garbage collection is off while ChimeraX starts and back on afterwards."""
import gc

import pytest

from chimerax.core import __main__ as main


@pytest.fixture
def restore_gc():
    yield
    gc.enable()
    gc.unfreeze()


def test_gc_is_off_during_startup_and_on_after(monkeypatch, restore_gc):
    seen = []
    monkeypatch.setattr(main, "_init", lambda argv, event_loop: seen.append(gc.isenabled()) or 0)
    assert main.init(["ChimeraX"]) == 0
    assert seen == [False]
    assert gc.isenabled()


def test_gc_is_on_after_failed_startup(monkeypatch, restore_gc):
    def fail(argv, event_loop):
        raise RuntimeError("startup failed")
    monkeypatch.setattr(main, "_init", fail)
    with pytest.raises(RuntimeError):
        main.init(["ChimeraX"])
    assert gc.isenabled()
