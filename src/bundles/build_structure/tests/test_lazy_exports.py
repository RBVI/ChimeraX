"""Registering Build Structure must not import its editing algorithms."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def fresh_bundle(monkeypatch):
    name = '_build_structure_lazy_exports_test'
    path = Path(__file__).parents[1] / 'src' / '__init__.py'
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module
    for key in list(sys.modules):
        if key.startswith(name + '.'):
            del sys.modules[key]


def test_import_and_discovery_leave_algorithms_unloaded(fresh_bundle):
    bundle = fresh_bundle
    assert bundle.__name__ + '.mod' not in sys.modules
    assert bundle.__name__ + '.start' not in sys.modules
    assert set(bundle._lazy_exports) <= set(dir(bundle))
    with pytest.raises(AttributeError, match='no attribute'):
        bundle.no_such_builder
    assert bundle.__name__ + '.mod' not in sys.modules


@pytest.mark.parametrize('name', ['modify_atom', 'cn_peptide_bond',
    'place_fragment', 'place_helium', 'place_peptide', 'PeptideError'])
def test_exports_keep_their_original_identity(fresh_bundle, name):
    from importlib import import_module
    bundle = fresh_bundle
    value = getattr(bundle, name)
    original = getattr(import_module('.' + bundle._lazy_exports[name], bundle.__name__), name)
    assert value is original
    assert bundle.__dict__[name] is original


def test_star_import_includes_functions_errors_and_provider_api(fresh_bundle):
    namespace = {}
    exec('from ' + fresh_bundle.__name__ + ' import *', namespace)
    assert namespace['modify_atom'] is fresh_bundle.mod.modify_atom
    assert namespace['PeptideError'] is fresh_bundle.start.PeptideError
    assert namespace['StartStructureProvider'] is fresh_bundle.provider_api.StartStructureProvider
    assert namespace['bundle_api'] is fresh_bundle.bundle_api
