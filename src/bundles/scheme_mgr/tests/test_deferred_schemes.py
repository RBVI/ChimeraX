from types import SimpleNamespace, MethodType


def test_schemes_wait_for_browser_then_register_once(monkeypatch):
    import sys
    import Qt
    from chimerax.scheme_mgr.manager import SchemesManager

    monkeypatch.delitem(sys.modules, 'Qt.QtWebEngineCore', raising=False)
    monkeypatch.setattr(Qt, 'qt_have_web_engine', lambda: True)
    registered = []
    manager = SimpleNamespace(
        schemes=set(), _pending_schemes=[], _web_engine_initialized=False,
        _register_scheme=lambda name, kw: registered.append((name, kw)),
    )
    manager.initialize_web_engine = MethodType(SchemesManager.initialize_web_engine, manager)
    SchemesManager.add_provider(manager, SimpleNamespace(installed=True), 'cxcmd', syntax='Path')
    assert manager.schemes == {'cxcmd'}
    assert not registered
    manager.initialize_web_engine()
    manager.initialize_web_engine()
    assert registered == [('cxcmd', {'syntax': 'Path'})]
    SchemesManager.add_provider(manager, SimpleNamespace(installed=True), 'new-scheme')
    assert registered[-1] == ('new-scheme', {})
    SchemesManager.add_provider(manager, SimpleNamespace(installed=False), 'not-installed')
    assert 'not-installed' not in manager.schemes


def test_unavailable_webengine_keeps_pending_schemes(monkeypatch):
    import Qt
    from chimerax.scheme_mgr.manager import SchemesManager

    monkeypatch.setattr(Qt, 'qt_have_web_engine', lambda: False)
    pending = [('help', {'syntax': 'Path'})]
    manager = SimpleNamespace(_pending_schemes=pending, _web_engine_initialized=False)
    SchemesManager.initialize_web_engine(manager)
    assert manager._pending_schemes == pending
    assert not manager._web_engine_initialized
