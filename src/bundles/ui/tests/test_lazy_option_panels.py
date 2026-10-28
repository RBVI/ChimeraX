from types import SimpleNamespace

import pytest


def test_option_panels_resolve_only_when_requested(monkeypatch):
    from chimerax.ui import options
    calls = []
    panel = object()
    module = SimpleNamespace(OptionsPanel=panel)
    for name in ('OptionsPanel', 'containers'):
        # Track initially absent exports too, so cached fakes are removed
        # during teardown rather than leaking into another test.
        monkeypatch.setitem(options.__dict__, name, object())
        monkeypatch.delitem(options.__dict__, name)

    def load(name, package):
        calls.append((name, package))
        return module

    monkeypatch.setattr('importlib.import_module', load)
    assert 'OptionsPanel' in dir(options)
    assert 'OptionsPanel' in options.__all__
    assert calls == []
    assert options.OptionsPanel is panel
    assert options.OptionsPanel is panel
    assert calls == [('.containers', options.__name__)]
    assert options.containers is module
    with pytest.raises(AttributeError):
        options.no_such_option_panel


def test_all_panel_exports_match_original_classes():
    from chimerax.ui import options
    from chimerax.ui.options import containers
    for name in options._container_names:
        assert getattr(options, name) is getattr(containers, name)
        assert name in options.__all__
    assert 'BooleanOption' in options.__all__
