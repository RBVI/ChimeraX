"""Optional atomic exports retain their identities and session class lookup."""
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def fresh_exports(monkeypatch):
    # Reuse the real data model to avoid registering a second set of Cython
    # classes, while executing the complete package initializer afresh.
    from chimerax import atomic
    name = '_atomic_exports_example'
    path = Path(__file__).parents[1] / 'src' / '__init__.py'
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    for submodule in ('molobject', 'pbgroup', 'molarray', 'structure', 'changes', 'triggers'):
        original = importlib.import_module('.' + submodule, atomic.__name__)
        monkeypatch.setitem(sys.modules, name + '.' + submodule, original)
    spec.loader.exec_module(module)
    return module


def test_discovery_and_unknown_names_do_not_import_implementations(monkeypatch, fresh_exports):
    atomic = fresh_exports
    calls = []
    monkeypatch.setattr(importlib, 'import_module', lambda *args: calls.append(args))
    assert set(atomic._lazy_exports) | atomic._lazy_modules <= set(dir(atomic))
    assert not set(atomic._lazy_exports) & vars(atomic).keys()
    with pytest.raises(AttributeError, match='no attribute'):
        atomic.no_such_atomic_type
    assert calls == []


def test_first_access_caches_the_original_object_and_supports_modules(monkeypatch, fresh_exports):
    atomic = fresh_exports
    original = object()
    implementation = SimpleNamespace(MolecularSurface=original)
    calls = []
    monkeypatch.setattr(importlib, 'import_module',
        lambda *args: calls.append(args) or implementation)
    assert atomic.MolecularSurface is original
    assert atomic.MolecularSurface is original
    assert calls == [('.molsurf', atomic.__name__)]
    assert atomic.molsurf is implementation
    assert atomic.molsurf is implementation
    assert len(calls) == 2


def test_exports_keep_identity_and_session_class_lookup():
    from chimerax import atomic
    for name, module_name in atomic._lazy_exports.items():
        implementation = importlib.import_module('.' + module_name, atomic.__name__)
        assert getattr(atomic, name) is getattr(implementation, name)
    for name in ('MolecularSurface', 'AtomicShapeDrawing', 'Atom', 'Structure'):
        assert atomic.bundle_api.get_class(name) is getattr(atomic, name)


def test_star_import_keeps_optional_objects_and_modules():
    from chimerax import atomic
    namespace = {}
    exec('from chimerax.atomic import *', namespace)
    for name in (*atomic._lazy_exports, *atomic._lazy_modules):
        assert namespace[name] is getattr(atomic, name)
