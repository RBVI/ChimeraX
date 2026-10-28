"""BundleInfo remembers where its package is instead of searching sys.path again."""
import importlib.util
from types import SimpleNamespace

from chimerax.core.toolshed.info import BundleInfo


def bundle(tmp_path, monkeypatch, package='example_pkg'):
    (tmp_path / 'icons').mkdir()
    (tmp_path / 'icons' / 'a.png').write_bytes(b'')
    calls = []
    def find_spec(name):
        calls.append(name)
        return SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(importlib.util, 'find_spec', find_spec)
    return BundleInfo('example', True, api_package_name=package), calls


def test_package_is_found_once_for_many_paths(tmp_path, monkeypatch):
    bi, calls = bundle(tmp_path, monkeypatch)
    assert bi.get_path('icons/a.png') == str(tmp_path / 'icons' / 'a.png')
    assert bi.get_path('icons/missing.png') is None
    assert bi.data_dir() is None
    assert calls == ['example_pkg']


def test_changed_package_name_or_unload_searches_again(tmp_path, monkeypatch):
    bi, calls = bundle(tmp_path, monkeypatch)
    bi.get_path('icons/a.png')
    bi.package_name = 'other_pkg'
    bi.get_path('icons/a.png')
    bi.unload(SimpleNamespace(info=lambda msg: None))
    bi.get_path('icons/a.png')
    assert calls == ['example_pkg', 'other_pkg', 'other_pkg']


def test_missing_package_is_not_remembered(tmp_path, monkeypatch):
    calls = []
    def find_spec(name):
        calls.append(name)
        if len(calls) == 1:
            raise ModuleNotFoundError(name)
        return SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(importlib.util, 'find_spec', find_spec)
    (tmp_path / 'data').mkdir()
    bi = BundleInfo('example', True, api_package_name='late_pkg')
    assert bi.get_path('data') is None
    assert bi.get_path('data') == str(tmp_path / 'data')
    assert calls == ['late_pkg', 'late_pkg']
