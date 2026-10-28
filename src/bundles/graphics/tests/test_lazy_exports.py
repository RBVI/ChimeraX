"""Importing drawing types need not load all renderers and camera variants."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def fresh_exports():
    path = Path(__file__).parents[1] / 'src' / '__init__.py'
    spec = importlib.util.spec_from_file_location('graphics_exports_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_and_discovery_leave_implementations_unloaded(monkeypatch):
    graphics = fresh_exports()
    calls = []
    drawing = object()
    monkeypatch.setattr(importlib, 'import_module',
        lambda name, package: calls.append((name, package)) or SimpleNamespace(Drawing=drawing))
    assert not any(name in vars(graphics) for name in graphics._exports)
    assert set(graphics._exports) <= set(dir(graphics))
    assert calls == []
    assert graphics.Drawing is drawing
    assert graphics.Drawing is drawing
    assert calls == [('.drawing', graphics.__name__)]
    assert 'OpenGLContext' not in vars(graphics)
    with pytest.raises(AttributeError):
        graphics.no_such_camera


def test_exports_and_session_classes_keep_original_identity():
    from chimerax import graphics
    for name, module_name in graphics._exports.items():
        implementation = importlib.import_module('.' + module_name, graphics.__name__)
        assert getattr(graphics, name) is getattr(implementation, name)
    for name in graphics.bundle_api._classes:
        assert graphics.bundle_api.get_class(name) is getattr(graphics, name)
    assert graphics.bundle_api.get_class('NoSuchGraphicsClass') is None


def test_star_import_keeps_existing_export_list():
    from chimerax import graphics
    namespace = {}
    exec('from chimerax.graphics import *', namespace)
    assert set(namespace) - {'__builtins__'} == set(graphics.__all__)
    for name in graphics.__all__:
        assert namespace[name] is getattr(graphics, name)
