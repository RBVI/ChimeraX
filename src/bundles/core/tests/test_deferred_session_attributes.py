"""Lazy bundle state has the same identity and serialization registration."""
import io

import pytest

from chimerax.core.session import Session
from chimerax.core.state import StateManager


class Manager(StateManager):
    def reset_state(self, session):
        pass


@pytest.fixture
def session():
    return Session(minimal=True)


def test_first_access_registers_real_state_manager_and_only_initializes_once(session):
    calls = []
    manager = Manager()
    def initialize(ses):
        calls.append(ses)
        ses.tracker = object()
        ses.manager = manager
    session._defer_attributes(('tracker', 'manager'), initialize)
    assert 'manager' in dir(session)
    assert 'manager' not in vars(session)
    assert session.get_state_manager('manager') is manager
    assert session.manager is manager
    assert session.tracker is vars(session)['tracker']
    assert calls == [session]
    assert not session._deferred_attributes


def test_manager_enumeration_resolves_pending_state(session):
    manager = Manager()
    session._defer_attributes(('manager',), lambda ses: setattr(ses, 'manager', manager))
    assert session.state_managers(Manager) == [manager]
    assert session.state_managers_by_tag(Manager) == {'manager': manager}


def test_failed_initializer_can_retry_without_replacing_completed_state(session):
    attempts = []
    manager = Manager()
    def initialize(ses):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError('unavailable')
        ses.manager = manager
    session._defer_attributes(('manager',), initialize)
    with pytest.raises(ValueError, match='unavailable'):
        session.manager
    assert session.manager is manager
    assert len(attempts) == 2


def test_explicit_assignment_cancels_pending_initializer(session):
    def unexpected(ses):
        raise AssertionError('explicit value must be retained')
    session._defer_attributes(('manager',), unexpected)
    manager = Manager()
    session.manager = manager
    assert session.get_state_manager('manager') is manager


def test_save_initializes_state_before_begin_save_handlers(session):
    manager = Manager()
    order = []
    def initialize(ses):
        order.append('initialize')
        ses.manager = manager
    session._defer_attributes(('manager',), initialize)
    class ReachedSave(Exception): pass
    def before_save(*args):
        assert session.manager is manager
        order.append('begin save')
        raise ReachedSave()
    # TriggerSet intentionally catches handler exceptions; use a direct trigger
    # facade to stop immediately after confirming serialization ordering.
    class Triggers:
        activate_trigger = staticmethod(before_save)
    session.triggers = Triggers()
    with pytest.raises(ReachedSave):
        session.save(io.BytesIO(), 3)
    assert order == ['initialize', 'begin save']


def test_missing_attribute_keeps_normal_attribute_error_and_dir(session):
    assert 'logger' in dir(session)
    assert not hasattr(session, 'unregistered')
    with pytest.raises(AttributeError):
        session.unregistered


def test_registration_conflicts_are_rejected(session):
    with pytest.raises(ValueError, match='already registered'):
        session._defer_attributes(('logger',), lambda ses: None)
    session._defer_attributes(('pending',), lambda ses: None)
    with pytest.raises(ValueError, match='already registered'):
        session._defer_attributes(('pending',), lambda ses: None)
    with pytest.raises(RuntimeError, match='did not assign'):
        session.pending


def test_concurrent_first_access_constructs_one_manager(session):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    entered = Event()
    release = Event()
    calls = []
    manager = Manager()
    def initialize(ses):
        calls.append(1)
        entered.set()
        assert release.wait(3)
        ses.manager = manager
    session._defer_attributes(('manager',), initialize)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(getattr, session, 'manager')
        assert entered.wait(3)
        second = pool.submit(getattr, session, 'manager')
        release.set()
        assert first.result() is second.result() is manager
    assert calls == [1]


def test_direct_registration_supersedes_a_deferred_manager(session):
    session._defer_attributes(('manager',), lambda ses: pytest.fail('unexpected initialization'))
    manager = Manager()
    session.add_state_manager('manager', manager)
    assert session.get_state_manager('manager') is manager
    assert not session._deferred_attributes
