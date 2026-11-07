"""Default sessions stay eager; deferred startup state completes only once."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from chimerax.core.session import Session


def test_default_construction_still_supplies_state_managers():
    session = Session()
    assert session.get_state_manager('main_view') is session.main_view
    assert session.models is not None and session.selection is not None
    assert session.user_colors and session.user_colormaps is not None


def test_deferred_state_is_complete_once_for_concurrent_callers(monkeypatch):
    session = Session(_defer_state=True)
    assert 'models' not in vars(session)
    assert session.logger and session.tasks is not None
    calls = []
    original = session._construct_state_managers
    def initialize():
        calls.append(True)
        original()
    monkeypatch.setattr(session, '_construct_state_managers', initialize)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: session._initialize_state(), range(4)))
    models = session.models
    session._initialize_state()
    assert session.models is models
    assert calls == [True]
    assert session.get_state_manager('main_view') is session.main_view


def test_failure_does_not_repeat_partial_state_registration(monkeypatch):
    session = Session(_defer_state=True)
    error = ValueError('unavailable')
    calls = []
    partial = {}
    def initialize():
        calls.append(True)
        session.save_options = partial
        raise error
    monkeypatch.setattr(session, '_construct_state_managers', initialize)
    for _ in range(2):
        with pytest.raises(ValueError) as caught:
            session._initialize_state()
        assert caught.value is error
        assert session.save_options is partial
    assert calls == [True]


def test_minimal_sessions_never_initialize_graphics_state(monkeypatch):
    session = Session(minimal=True, _defer_state=True)
    monkeypatch.setattr(session, '_construct_state_managers', lambda: pytest.fail('Initialized minimal session'))
    session._initialize_state()
    assert 'models' not in vars(session)
