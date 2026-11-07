"""Native menus remain on main; optional definition imports always join."""
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from chimerax.ui import native_menu as native


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(native, 'sys', SimpleNamespace(platform='darwin',
        implementation=SimpleNamespace(name='cpython'), modules={}))
    class MainWindow:
        __module__ = 'chimerax.ui.gui'
        def menuBar(self):
            return self.bar
    window = MainWindow()
    window.bar = object()
    window.session = SimpleNamespace(ui=SimpleNamespace(_native_qapplication=True,
        autostart_tools=True, settings=SimpleNamespace(autostart=['Log'])))
    return window


def test_module_selection_respects_tools_and_existing_imports(window):
    ui = window.session.ui
    assert native._definition_modules(ui) == ('chimerax.log.parse', 'chimerax.log.view')
    native.sys.modules['chimerax.log.parse'] = object()
    assert native._definition_modules(ui) == ('chimerax.log.view',)
    ui.settings.autostart = ['Other tool']
    assert native._definition_modules(ui) == ()
    ui.settings.autostart = ['Log']
    ui.autostart_tools = False
    assert native._definition_modules(ui) == ()


def test_import_failure_leaves_later_imports_available_for_normal_startup(monkeypatch):
    calls = []
    def import_module(name):
        calls.append(name)
        if name == 'broken':
            raise ImportError('unavailable')
    monkeypatch.setattr(native, 'import_module', import_module)
    native._import_definitions(('first', 'broken', 'last'))
    assert calls == ['first', 'broken', 'last']


def test_native_creation_is_on_main_and_worker_finishes_before_return(window, monkeypatch):
    main = get_ident()
    native_started, worker_started, worker_done = Event(), Event(), Event()
    calls = []
    def definitions(names):
        assert get_ident() != main
        worker_started.set()
        assert native_started.wait(5)
        worker_done.set()
    def construct():
        assert get_ident() == main
        native_started.set()
        assert worker_started.wait(5)
        calls.append('native')
    def leave(context):
        assert worker_done.is_set()
        calls.append('leave')
    monkeypatch.setattr(native, '_import_definitions', definitions)
    monkeypatch.setattr(native, '_prepare_menu_bar', lambda window:
        (construct, lambda context: calls.append('enter'), leave))
    assert native.menu_bar(window) is window.bar
    assert worker_done.is_set() and window._native_menu_bar_overlap
    assert calls == ['enter', 'native', 'leave']


def test_native_failure_joins_worker_and_leaves_context(window, monkeypatch):
    done = Event()
    error = ValueError('native callback')
    def fail():
        raise error
    def leave(context):
        assert done.is_set()
    monkeypatch.setattr(native, '_import_definitions', lambda modules: done.set())
    monkeypatch.setattr(native, '_prepare_menu_bar', lambda window:
        (fail, lambda context: None, leave))
    with pytest.raises(ValueError) as caught:
        native.menu_bar(window)
    assert caught.value is error and done.is_set()


def test_thread_exhaustion_uses_normal_menu_and_closes_context(window, monkeypatch):
    calls = []
    class UnavailableThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            raise RuntimeError('unavailable')
        def join(self):
            pytest.fail('Joined a thread that never started')
    monkeypatch.setattr(native, 'Thread', UnavailableThread)
    monkeypatch.setattr(native, '_prepare_menu_bar', lambda window:
        (lambda: pytest.fail('Called native menu'), lambda context: calls.append('enter'),
            lambda context: calls.append('leave')))
    assert native.menu_bar(window) is window.bar
    assert calls == ['enter', 'leave']


@pytest.mark.parametrize('error', [ValueError('image'), OSError('library'), AttributeError('symbol')])
def test_unavailable_native_api_falls_back_before_starting_a_worker(window, monkeypatch, error):
    def prepare(window):
        raise error
    monkeypatch.setattr(native, '_prepare_menu_bar', prepare)
    monkeypatch.setattr(native, 'Thread', lambda **kw: pytest.fail('Started a worker'))
    assert native.menu_bar(window) is window.bar


def test_memory_error_is_not_masked(window, monkeypatch):
    def prepare(window):
        raise MemoryError('allocation')
    monkeypatch.setattr(native, '_prepare_menu_bar', prepare)
    with pytest.raises(MemoryError):
        native.menu_bar(window)


@pytest.mark.parametrize('case', ['linux', 'win32', 'interpreter', 'thread', 'binding',
    'subclass', 'no_tools', 'custom_tools', 'already_loaded'])
def test_unsupported_or_unnecessary_preload_uses_ordinary_binding(window, monkeypatch, case):
    if case in ('linux', 'win32'):
        native.sys.platform = case
    elif case == 'interpreter':
        native.sys.implementation.name = 'pypy'
    elif case == 'thread':
        monkeypatch.setattr(native, 'current_thread', lambda: object())
    elif case == 'binding':
        window.session.ui._native_qapplication = False
    elif case == 'subclass':
        class OtherWindow(type(window)):
            pass
        window.__class__ = OtherWindow
    elif case == 'no_tools':
        window.session.ui.autostart_tools = False
    elif case == 'custom_tools':
        window.session.ui.settings.autostart = ['Custom tool']
    else:
        native.sys.modules.update(dict.fromkeys(('chimerax.log.parse', 'chimerax.log.view')))
    monkeypatch.setattr(native, '_prepare_menu_bar', lambda window: pytest.fail('Prepared native API'))
    assert native.menu_bar(window) is window.bar
