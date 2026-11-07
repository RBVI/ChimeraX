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


@pytest.mark.parametrize(('name', 'module_name'), [
    ('Drawing', 'drawing'),
    ('image_from_rgba', 'view'),
])
def test_import_and_discovery_leave_implementations_unloaded(monkeypatch, name, module_name):
    graphics = fresh_exports()
    calls = []
    exported = object()
    monkeypatch.setattr(importlib, 'import_module',
        lambda module, package: calls.append((module, package)) or SimpleNamespace(**{name: exported}))
    assert not any(name in vars(graphics) for name in graphics._exports)
    assert set(graphics._exports) <= set(dir(graphics))
    assert calls == []
    assert getattr(graphics, name) is exported
    assert getattr(graphics, name) is exported
    assert calls == [('.' + module_name, graphics.__name__)]
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
