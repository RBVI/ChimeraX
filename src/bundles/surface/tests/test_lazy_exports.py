"""Surface exports stay discoverable without loading unused algorithms."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def fresh_exports():
    path = Path(__file__).parents[1] / 'src' / '__init__.py'
    spec = importlib.util.spec_from_file_location('surface_exports_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_is_lazy_and_cached(monkeypatch):
    surface = fresh_exports()
    assert not any(name in vars(surface) for name in surface._surface_exports)
    assert not any(name in vars(surface) for name in surface._surface_modules)
    function = object()
    calls = []

    def load(name, package):
        calls.append((name, package))
        return SimpleNamespace(sphere_geometry=function)

    monkeypatch.setattr(importlib, 'import_module', load)
    assert 'sphere_geometry' in dir(surface)
    assert 'sphere_geometry' in surface.__all__
    assert surface.sphere_geometry is function
    assert surface.sphere_geometry is function
    assert calls == [('.shapes', surface.__name__)]
    assert 'color_radial' not in vars(surface)
    with pytest.raises(AttributeError):
        surface.not_a_surface_export


def test_module_exports_and_wildcard_import(monkeypatch):
    surface = fresh_exports()
    implementations = {}
    for name, module_name in surface._surface_exports.items():
        module = implementations.setdefault(module_name, SimpleNamespace())
        setattr(module, name, object())
    monkeypatch.setattr(importlib, 'import_module',
                        lambda name, package: implementations[name[1:]])
    # __all__ drives wildcard imports; every exported name must resolve.
    exported = {name: getattr(surface, name) for name in surface.__all__}
    assert exported['shapes'] is implementations['shapes']
    assert exported['sphere_geometry'] is implementations['shapes'].sphere_geometry
    for name, module_name in surface._surface_exports.items():
        assert exported[name] is getattr(implementations[module_name], name)


def test_public_exports_are_original_implementation_objects():
    from chimerax import surface
    for name, module_name in surface._surface_exports.items():
        module = importlib.import_module('.' + module_name, surface.__name__)
        assert getattr(surface, name) is getattr(module, name)
