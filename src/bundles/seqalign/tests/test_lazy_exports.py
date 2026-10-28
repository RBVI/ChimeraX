"""Public imports and provider annotations must survive deferred command loading."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def fresh_bundle(monkeypatch):
    name = '_seqalign_lazy_exports_test'
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


def test_import_and_discovery_do_not_load_commands_or_alignments(fresh_bundle):
    bundle = fresh_bundle
    assert bundle.__name__ + '.cmd' not in sys.modules
    assert bundle.__name__ + '.alignment' not in sys.modules
    assert {'SeqArg', 'AlignmentArg', 'clustal_strong_groups', 'cmd', 'alignment'} <= set(dir(bundle))
    with pytest.raises(AttributeError, match='no attribute'):
        bundle.no_such_annotation
    assert bundle.__name__ + '.cmd' not in sys.modules
    assert bundle.__name__ + '.alignment' not in sys.modules


def test_api_and_package_return_the_original_annotation(fresh_bundle):
    bundle = fresh_bundle
    argument = type(bundle.bundle_api).AlignmentArg
    assert argument is bundle.bundle_api.AlignmentArg
    assert argument is bundle.AlignmentArg is bundle.cmd.AlignmentArg
    alignment = SimpleNamespace(ident='example')
    session = SimpleNamespace(alignments=SimpleNamespace(
        alignments=[alignment], alignments_map={'example': alignment}))
    parsed, _, rest = argument.parse('example trailing', session)
    assert parsed is alignment and rest == 'trailing'


def test_provider_argument_factories_load_on_first_use(fresh_bundle):
    bundle = fresh_bundle
    session = SimpleNamespace(open_command=object(), alignments=SimpleNamespace(
        viewer_info={'alignment': {}, 'sequence': {}}))
    opener = bundle.bundle_api.run_provider(session, 'fasta', session.open_command)
    saver = bundle.bundle_api.run_provider(session, 'fasta', object())
    assert bundle.__name__ + '.cmd' not in sys.modules
    assert set(opener.open_args) == {'alignment', 'auto_associate', 'ident', 'viewer'}
    assert saver.save_args['alignment'] is bundle.AlignmentArg


def test_star_import_preserves_exports_and_identity(fresh_bundle):
    namespace = {}
    exec('from ' + fresh_bundle.__name__ + ' import *', namespace)
    assert namespace['SeqArg'] is fresh_bundle.cmd.SeqArg
    assert namespace['clustal_strong_groups'] is fresh_bundle.alignment.clustal_strong_groups
    assert namespace['cmd'] is fresh_bundle.cmd
    assert namespace['alignment'] is fresh_bundle.alignment
    assert namespace['manager'] is fresh_bundle.manager
    assert namespace['settings'] is fresh_bundle.settings
    assert namespace['bundle_api'].get_class('Alignment') is fresh_bundle.alignment.Alignment
