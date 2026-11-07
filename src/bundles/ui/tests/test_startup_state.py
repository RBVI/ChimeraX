"""Session work is joined before settings, including failed GUI construction."""
from threading import Event, get_ident
from types import SimpleNamespace

import pytest


@pytest.fixture
def startup(monkeypatch):
    from chimerax.ui import startup
    return startup


def test_state_overlaps_native_work_and_finishes_before_settings(startup, monkeypatch):
    main = get_ident()
    native_started, state_started, state_done = Event(), Event(), Event()
    calls = []
    def state():
        assert get_ident() != main
        state_started.set()
        assert native_started.wait(5)
        state_done.set()
    def native():
        assert get_ident() == main
        native_started.set()
        assert state_started.wait(5)
    def construct():
        assert get_ident() == main
        calls.append('application')
    def settings():
        assert state_done.is_set()
        assert calls == ['application', 'text', 'close']
        return 'settings'
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: SimpleNamespace(
        initialize=native, warm_text_system=lambda: calls.append('text'), close=lambda: calls.append('close')))
    assert startup.initialize_application(SimpleNamespace(_initialize_state=state), construct, settings) == 'settings'


@pytest.mark.parametrize('state_fails', [False, True])
def test_failed_application_still_joins_state_and_closes_native(startup, monkeypatch, state_fails):
    state_done = Event()
    calls = []
    state_error, application_error = TypeError('state'), ValueError('application')
    def state():
        state_done.set()
        if state_fails:
            raise state_error
    def construct():
        raise application_error
    def close():
        assert state_done.is_set()
        calls.append('close')
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: SimpleNamespace(
        initialize=lambda: None, warm_text_system=lambda: None, close=close))
    with pytest.raises(ValueError) as caught:
        startup.initialize_application(SimpleNamespace(_initialize_state=state), construct,
            lambda: pytest.fail('Settings ran after application failure'))
    assert caught.value is application_error
    assert caught.value.__cause__ is (state_error if state_fails else None)
    assert calls == ['close']


def test_worker_error_is_raised_on_caller_before_settings(startup, monkeypatch):
    error = ValueError('state')
    calls = []
    def state():
        raise error
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: SimpleNamespace(
        initialize=lambda: None, warm_text_system=lambda: None, close=lambda: calls.append('close')))
    with pytest.raises(ValueError) as caught:
        startup.initialize_application(SimpleNamespace(_initialize_state=state), lambda: None,
            lambda: pytest.fail('Settings ran after worker failure'))
    assert caught.value is error
    assert calls == ['close']


def test_thread_exhaustion_falls_back_to_synchronous_state(startup, monkeypatch):
    calls = []
    class UnavailableThread:
        def __init__(self, **kw):
            pass
        def start(self):
            raise RuntimeError('No thread available')
        def join(self):
            pytest.fail('Joined a thread that never started')
    monkeypatch.setattr(startup, 'Thread', UnavailableThread)
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: SimpleNamespace(
        initialize=lambda: calls.append('native'), warm_text_system=lambda: calls.append('text'), close=lambda: calls.append('close')))
    result = startup.initialize_application(
        SimpleNamespace(_initialize_state=lambda: calls.append('state')),
        lambda: calls.append('application'), lambda: calls.append('settings'))
    assert calls == ['state', 'native', 'application', 'text', 'close', 'settings']


def test_unavailable_native_path_preserves_synchronous_order(startup, monkeypatch):
    calls = []
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: None)
    startup.initialize_application(SimpleNamespace(_initialize_state=lambda: calls.append('state')),
        lambda: calls.append('application'), lambda: calls.append('settings'))
    assert calls == ['state', 'application', 'settings']


def test_settings_failure_occurs_after_native_cleanup(startup, monkeypatch):
    closed = []
    def settings():
        assert closed == [True]
        raise ValueError('settings')
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: SimpleNamespace(
        initialize=lambda: None, warm_text_system=lambda: None, close=lambda: closed.append(True)))
    with pytest.raises(ValueError, match='settings'):
        startup.initialize_application(SimpleNamespace(_initialize_state=lambda: None), lambda: None, settings)


def test_thread_allocation_failure_closes_the_prepared_pool(startup, monkeypatch):
    closed = []
    def unavailable(**kwargs):
        raise MemoryError('thread allocation')
    monkeypatch.setattr(startup, 'Thread', unavailable)
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: SimpleNamespace(
        initialize=lambda: pytest.fail('Native initialization ran'), close=lambda: closed.append(True)))
    with pytest.raises(MemoryError):
        startup.initialize_application(SimpleNamespace(_initialize_state=lambda: None), lambda: None, lambda: None)
    assert closed == [True]


@pytest.mark.parametrize('platform', ['linux', 'win32'])
def test_other_platforms_never_prepare_appkit(startup, monkeypatch, platform):
    monkeypatch.setattr(startup, 'sys', SimpleNamespace(platform=platform))
    monkeypatch.setattr(startup, '_NativeApplication', lambda: pytest.fail('Loaded AppKit'))
    assert startup._prepare_native_application() is None


def test_existing_native_application_is_left_to_qt(startup, monkeypatch):
    closed = []
    monkeypatch.setattr(startup, 'sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(startup, '_NativeApplication', lambda: SimpleNamespace(
        existing=True, close=lambda: closed.append(True)))
    assert startup._prepare_native_application() is None
    assert closed == [True]


def test_missing_framework_preserves_normal_startup(startup, monkeypatch):
    monkeypatch.setattr(startup, 'sys', SimpleNamespace(platform='darwin'))
    def unavailable():
        raise OSError('framework unavailable')
    monkeypatch.setattr(startup, '_NativeApplication', unavailable)
    assert startup._prepare_native_application() is None


def test_text_warmup_runs_on_main_thread_while_state_can_finish(startup, monkeypatch):
    main = get_ident()
    text_started, state_done = Event(), Event()
    calls = []
    def state():
        assert text_started.wait(5)
        state_done.set()
    def text():
        assert get_ident() == main
        assert calls == ['application']
        text_started.set()
        assert state_done.wait(5)
        calls.append('text')
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: SimpleNamespace(
        initialize=lambda: None, warm_text_system=text, close=lambda: calls.append('close')))
    startup.initialize_application(SimpleNamespace(_initialize_state=state),
        lambda: calls.append('application'), lambda: calls.append('settings'))
    assert calls == ['application', 'text', 'close', 'settings']


@pytest.mark.parametrize('failure', [None, 'class', 'selector', 'method'])
def test_optional_text_system_uses_only_an_available_class_method(startup, failure):
    calls = []
    def get_class(name):
        assert name == b'NSWritingToolsCoordinator'
        return None if failure == 'class' else 17
    def selector(name):
        assert name == b'isWritingToolsAvailable'
        return None if failure == 'selector' else 23
    def get_method(cls, sel):
        assert (cls, sel) == (17, 23)
        return None if failure == 'method' else 31
    def send(cls, sel):
        calls.append((cls, sel))
        return False
    initialize = startup._text_system_initializer(get_class, selector, get_method, send)
    assert not calls
    if failure is None:
        assert initialize() is False
        assert calls == [(17, 23)]
    else:
        assert initialize is None


def test_failed_text_warmup_joins_worker_and_preserves_the_error(startup, monkeypatch):
    done = Event()
    closed = []
    error = ValueError('text setup')
    def fail():
        raise error
    def close():
        assert done.is_set()
        closed.append(True)
    monkeypatch.setattr(startup, '_prepare_native_application', lambda: SimpleNamespace(
        initialize=lambda: None, warm_text_system=fail, close=close))
    with pytest.raises(ValueError) as caught:
        startup.initialize_application(SimpleNamespace(_initialize_state=done.set), lambda: None,
            lambda: pytest.fail('Settings ran after native failure'))
    assert caught.value is error
    assert closed == [True]
