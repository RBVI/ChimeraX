"""Bundle registration keeps molecular implementations lazy."""
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def exports(monkeypatch):
    path = Path(__file__).parents[1] / 'src' / '__init__.py.in'
    name = '_chem_group_registration'
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)])
    # The chemical-group template uses a .in suffix.
    if spec is None:
        from importlib.machinery import SourceFileLoader
        spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)), is_package=True)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_registration_and_discovery_do_not_load_implementations(exports):
    assert exports._lazy_exports
    assert not set(exports._lazy_exports).intersection(vars(exports))
    assert set(exports._lazy_exports) <= set(dir(exports))
    assert exports.__name__ + '.chem_group' not in sys.modules
    with pytest.raises(AttributeError, match='no attribute'):
        exports.unknown_export


def test_exports_retain_actual_objects_and_cache_their_identity():
    bundle = importlib.import_module('chimerax.chem_group')
    for name in bundle._lazy_exports:
        module_name = 'chem_group'
        implementation = importlib.import_module('.' + module_name, bundle.__name__)
        assert getattr(bundle, name) is getattr(implementation, name)
        assert vars(bundle)[name] is getattr(bundle, name)
    namespace = {}
    exec('from chimerax.chem_group import *', namespace)
    for name in (*bundle._lazy_exports, *bundle._lazy_modules):
        assert namespace[name] is getattr(bundle, name)
