"""Manager state can overlap native work without moving GUI work off main."""
import sys
from importlib.machinery import SourceFileLoader
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from chimerax.ui import manager_startup as startup
from chimerax.ui import native_menu


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    from chimerax import core
    from chimerax.core.toolshed.info import BundleInfo
    monkeypatch.setattr(core, '__file__', str(tmp_path / 'chimerax/core/__init__.py'))
    bi = BundleInfo.__new__(BundleInfo)
    bi._name = 'ChimeraX-PresetMgr'
    bi.package_name = 'chimerax.preset_mgr'
    bi.managers = {'presets': {}}
    bi.providers = {}
    toolshed = SimpleNamespace(_manager_instances={}, _installed_bundle_info=[bi])
    def spec(name):
        path = str(tmp_path / 'chimerax' / name.split('.')[-1] / '__init__.py')
        return SimpleNamespace(origin=path, loader=SourceFileLoader(name, path))
    monkeypatch.setattr(startup, 'find_spec', spec)
    for name in ('Qt.QtWebEngineCore', 'PySide6.QtWebEngineCore'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return toolshed, bi


def test_only_active_bundled_managers_are_eligible(catalog):
    toolshed, bi = catalog
    assert startup._bundled_managers(toolshed)
    bi.managers['custom manager'] = {'autostart': 'false'}
    assert startup._bundled_managers(toolshed)
    bi.managers['custom manager']['autostart'] = 'true'
    assert not startup._bundled_managers(toolshed)


@pytest.mark.parametrize('case', ['name', 'package', 'origin', 'loader', 'missing',
    'import_error', 'already_initialized', 'eager_provider', 'web_engine'])
def test_unsafe_manager_catalogs_use_main_thread(catalog, monkeypatch, case):
    toolshed, bi = catalog
    if case == 'name':
        bi._name = 'Custom-PresetMgr'
    elif case == 'package':
        bi.package_name = 'custom.preset_mgr'
    elif case == 'origin':
        monkeypatch.setattr(startup, 'find_spec', lambda name: SimpleNamespace(
            origin='/user/preset_mgr/__init__.py', loader=SourceFileLoader(name, 'user')))
    elif case == 'loader':
        monkeypatch.setattr(startup, 'find_spec', lambda name: SimpleNamespace(
            origin='__init__.py', loader=object()))
    elif case == 'missing':
        monkeypatch.setattr(startup, 'find_spec', lambda name: None)
    elif case == 'import_error':
        def fail(name):
            raise ImportError('unavailable')
        monkeypatch.setattr(startup, 'find_spec', fail)
    elif case == 'already_initialized':
        toolshed._manager_instances['existing'] = object()
    elif case == 'eager_provider':
        bi.providers['segmentation formats/custom'] = {}
    else:
        monkeypatch.setitem(sys.modules, 'Qt.QtWebEngineCore', object())
    assert not startup._bundled_managers(toolshed)


@pytest.fixture
def overlap(monkeypatch):
    main = get_ident()
    native_started, managers_done, definitions_done = Event(), Event(), Event()
    calls = []
    class Window:
        def menuBar(self):
            assert get_ident() == main
            calls.append('wrapper')
    def managers(session):
        assert get_ident() != main
        assert native_started.wait(5)
        managers_done.set()
        return ['failed bundle']
    def definitions(modules):
        assert get_ident() != main and managers_done.is_set()
        assert modules == ('definitions',)
        definitions_done.set()
    def construct():
        assert get_ident() == main
        calls.append('native')
        native_started.set()
        assert definitions_done.wait(5)
    def leave(context):
        assert get_ident() == main and definitions_done.is_set()
        calls.append('leave')
    monkeypatch.setattr(native_menu, '_definition_modules', lambda ui: ('definitions',))
    monkeypatch.setattr(native_menu, '_import_definitions', definitions)
    monkeypatch.setattr(native_menu, '_prepare_menu_bar', lambda window:
        (construct, lambda context: calls.append('enter'), leave))
    return Window(), SimpleNamespace(_init_managers=managers), SimpleNamespace(ui=object()), calls


def test_managers_and_definitions_finish_before_return_and_preserve_failures(overlap):
    window, toolshed, session, calls = overlap
    assert startup._overlap(window, toolshed, session) == ['failed bundle']
    assert calls == ['enter', 'native', 'wrapper', 'leave']
    assert window._native_menu_bar_overlap


def test_native_error_waits_for_worker_and_leaves_context(monkeypatch):
    done = Event()
    error = ValueError('native')
    calls = []
    def fail():
        raise error
    def leave(context):
        assert done.is_set()
        calls.append('leave')
    monkeypatch.setattr(native_menu, '_prepare_menu_bar', lambda window:
        (fail, lambda context: None, leave))
    monkeypatch.setattr(native_menu, '_definition_modules', lambda ui: ())
    monkeypatch.setattr(native_menu, '_import_definitions', lambda modules: None)
    def managers(session):
        done.set()
        return []
    with pytest.raises(ValueError) as caught:
        startup._overlap(object(), SimpleNamespace(_init_managers=managers),
                         SimpleNamespace(ui=object()))
    assert caught.value is error and calls == ['leave']


def test_worker_error_is_reraised_on_main_after_context_cleanup(monkeypatch):
    error = LookupError('manager')
    calls = []
    def managers(session):
        raise error
    monkeypatch.setattr(native_menu, '_prepare_menu_bar', lambda window:
        (lambda: None, lambda context: None, lambda context: calls.append('leave')))
    monkeypatch.setattr(native_menu, '_definition_modules', lambda ui: ())
    monkeypatch.setattr(native_menu, '_import_definitions', lambda modules:
        pytest.fail('Imported definitions after manager failure'))
    with pytest.raises(LookupError) as caught:
        startup._overlap(SimpleNamespace(menuBar=lambda: None),
            SimpleNamespace(_init_managers=managers), SimpleNamespace(ui=object()))
    assert caught.value is error and calls == ['leave']


def test_thread_exhaustion_returns_to_sequential_initialization(monkeypatch):
    calls = []
    class UnavailableThread:
        def __init__(self, **kw):
            pass
        def start(self):
            raise RuntimeError('No threads')
        def join(self):
            pytest.fail('Joined an unstarted worker')
    monkeypatch.setattr(startup, 'Thread', UnavailableThread)
    monkeypatch.setattr(native_menu, '_prepare_menu_bar', lambda window:
        (lambda: pytest.fail('Created native menu'), lambda context: calls.append('enter'),
         lambda context: calls.append('leave')))
    monkeypatch.setattr(native_menu, '_definition_modules', lambda ui: ())
    assert startup._overlap(object(), object(), SimpleNamespace(ui=object())) is None
    assert calls == ['enter', 'leave']


@pytest.fixture
def initialization(monkeypatch):
    from chimerax.core.toolshed import Toolshed
    calls = []
    class UI:
        _native_qapplication = True
        main_window = None
    class MainWindow:
        def __init__(self, ui, session, *, _defer_build=False):
            assert _defer_build
            calls.append('native window')
    ui = UI()
    session = SimpleNamespace(ui=ui)
    toolshed = Toolshed.__new__(Toolshed)
    toolshed._init_managers = lambda session: calls.append('sequential') or []
    monkeypatch.setattr(startup, 'sys', SimpleNamespace(platform='darwin',
        implementation=SimpleNamespace(name='cpython')))
    monkeypatch.setitem(sys.modules, 'chimerax.ui.gui', SimpleNamespace(UI=UI, MainWindow=MainWindow))
    monkeypatch.setitem(sys.modules, 'shiboken6', SimpleNamespace(delete=lambda window:
        calls.append('delete window')))
    monkeypatch.setattr(startup, '_bundled_managers', lambda toolshed: True)
    return toolshed, session, calls


def test_prepared_window_is_retained_until_normal_ui_build(initialization, monkeypatch):
    toolshed, session, calls = initialization
    monkeypatch.setattr(startup, '_overlap', lambda *args: ['failed bundle'])
    assert startup.initialize_managers(toolshed, session) == ['failed bundle']
    assert session.ui.main_window is None
    assert session.ui._prepared_main_window is not None
    assert calls == ['native window']


def test_native_fallback_destroys_unused_window_before_sequential_managers(initialization, monkeypatch):
    toolshed, session, calls = initialization
    monkeypatch.setattr(startup, '_overlap', lambda *args: None)
    assert startup.initialize_managers(toolshed, session) == []
    assert calls == ['native window', 'delete window', 'sequential']
    assert not hasattr(session.ui, '_prepared_main_window')


def test_failed_overlap_destroys_unused_window_without_repeating_managers(initialization, monkeypatch):
    toolshed, session, calls = initialization
    error = ValueError('manager failed')
    def fail(*args):
        raise error
    monkeypatch.setattr(startup, '_overlap', fail)
    with pytest.raises(ValueError) as caught:
        startup.initialize_managers(toolshed, session)
    assert caught.value is error
    assert calls == ['native window', 'delete window']
    assert not hasattr(session.ui, '_prepared_main_window')


@pytest.mark.parametrize('case', ['linux', 'interpreter', 'thread', 'binding', 'existing', 'pending', 'catalog'])
def test_ineligible_startup_does_not_construct_a_window(initialization, monkeypatch, case):
    toolshed, session, calls = initialization
    if case == 'linux':
        startup.sys.platform = 'linux'
    elif case == 'interpreter':
        startup.sys.implementation.name = 'pypy'
    elif case == 'thread':
        monkeypatch.setattr(startup, 'current_thread', lambda: object())
    elif case == 'binding':
        session.ui._native_qapplication = False
    elif case == 'existing':
        session.ui.main_window = object()
    elif case == 'pending':
        session.ui._prepared_main_window = object()
    else:
        monkeypatch.setattr(startup, '_bundled_managers', lambda toolshed: False)
    assert startup.initialize_managers(toolshed, session) == []
    assert calls == ['sequential']


def test_ui_build_finishes_and_reuses_the_prepared_window():
    from chimerax.ui.gui import UI
    calls = []
    session = object()
    window = SimpleNamespace(_finish_build=lambda ui, ses: calls.append((ui, ses)))
    ui = SimpleNamespace(session=session, _prepared_main_window=window)
    assert UI._create_main_window(ui) is window
    assert calls == [(ui, session)]
    assert not hasattr(ui, '_prepared_main_window')


def test_ui_build_without_prepared_window_keeps_normal_construction(monkeypatch):
    from chimerax.ui import gui
    calls = []
    window = object()
    monkeypatch.setattr(gui, 'MainWindow', lambda ui, session:
        calls.append((ui, session)) or window)
    ui = SimpleNamespace(session=object())
    assert gui.UI._create_main_window(ui) is window
    assert calls == [(ui, ui.session)]
