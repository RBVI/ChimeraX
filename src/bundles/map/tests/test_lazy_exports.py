"""Map startup registration need not import volume commands or algorithms."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def fresh_exports(name='map_exports_example'):
    path = Path(__file__).parents[1] / 'src' / '__init__.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_leaves_volume_types_and_commands_unloaded(monkeypatch):
    maps = fresh_exports()
    calls = []
    original = object()
    implementation = SimpleNamespace(Volume=original)
    monkeypatch.setattr(importlib, 'import_module', lambda *args: calls.append(args) or implementation)
    assert not set(maps._lazy_exports) & vars(maps).keys()
    assert set(maps._lazy_exports) <= set(dir(maps))
    with pytest.raises(AttributeError, match='no attribute'):
        maps.NoSuchMap
    assert calls == []
    assert maps.Volume is original
    assert maps.Volume is original
    assert calls == [('.volume', maps.__name__)]


def test_exports_keep_original_identity_and_session_classes():
    from chimerax import map as maps
    for name, module_name in maps._lazy_exports.items():
        assert getattr(maps, name) is getattr(
            importlib.import_module('.' + module_name, maps.__name__), name)
    for name in ('Volume', 'VolumeSurface', 'VolumeImage', 'MapChannelsModel', 'MultiChannelSeries'):
        assert maps.bundle_api.get_class(name) is getattr(maps, name)


def test_saver_widget_resolves_volume_without_prior_public_access(monkeypatch):
    from chimerax import map as maps
    from chimerax.map.volume import Volume
    import chimerax.save_command.widgets as widgets

    fresh = fresh_exports(maps.__name__)
    # Use a real package for the method's relative imports, with no Volume
    # global cached in the freshly loaded package implementation.
    created = []
    monkeypatch.setattr(widgets, 'SaveModelOptionWidget',
        lambda *args: created.append(args) or 'widget')
    session = SimpleNamespace(open_command=object())
    saver = fresh.bundle_api.run_provider(session, 'MRC density map', object())
    assert 'Volume' not in vars(fresh)
    assert saver.save_args_widget(session) == 'widget'
    assert created == [(session, 'Map', Volume)]


def test_star_import_preserves_public_objects_and_submodules():
    from chimerax import map as maps
    namespace = {}
    exec('from chimerax.map import *', namespace)
    for name in (*maps._lazy_exports, *maps._lazy_modules):
        assert namespace[name] is getattr(maps, name)
