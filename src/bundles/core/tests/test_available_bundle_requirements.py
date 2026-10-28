import pytest
from packaging.version import Version
from packaging.requirements import Requirement

from chimerax.core.toolshed import available


def bundle(requirements):
    return available._build_bundle({
        'bundle_name': 'ChimeraX_Example',
        'bundle': {'ChimeraX_Example': {'version': '1', 'requires': requirements}},
    })


def test_compatibility_only_parses_prefix_and_full_list_reuses_objects(monkeypatch):
    calls = []
    parse = available._parse_requires
    monkeypatch.setattr(available, '_parse_requires',
                        lambda text: calls.append(text) or parse(text))
    texts = ['invalid requirement!', 'numpy', 'ChimeraX-Core>=1', 'other>=2']
    info = bundle(texts)
    assert calls == []
    cache = available.AvailableBundleCache(None)
    assert cache._installable(info, Version('1.5'))
    assert calls == texts[:3]
    core = info._parsed_requirements[2]
    requirements = info.requires
    assert calls == texts
    assert requirements[1] is core
    assert [str(r) for r in requirements] == texts[1:]
    assert info.requires is requirements


@pytest.mark.parametrize('texts,expected', [
    ([], False), (['numpy'], False),
    (['ChimeraX-Core>=2'], False),
    (['ChimeraX-Core>=1; python_version < "1"'], False),
    (['ChimeraX-Core>=1; python_version >= "3"'], True),
    (['ChimeraX-Core>=2', 'ChimeraX-Core>=1'], False),
    (['ChimeraX-Core>=1.5.dev0'], True),
])
@pytest.mark.parametrize('checks', [None, {}])
def test_compatibility_preserves_markers_order_and_prereleases(texts, expected, checks):
    assert available.AvailableBundleCache(None)._installable(
        bundle(texts), Version('1.5.dev1'), None if checks is None else {}) is expected


def test_mutable_requirements_remain_independent_between_bundles():
    first = bundle(['ChimeraX-Core>=1'])
    second = bundle(['ChimeraX-Core>=1'])
    first.requires[0].name = 'different'
    assert second.requires[0].name == 'ChimeraX-Core'
    cache = available.AvailableBundleCache(None)
    assert not cache._installable(first, Version('1.5'))
    first.requires = [Requirement('ChimeraX-Core>=1')]
    assert cache._installable(first, Version('1.5'))
    first.requires.clear()
    assert not cache._installable(first, Version('1.5'))


def test_catalog_reuses_decisions_without_sharing_requirements(monkeypatch):
    import chimerax.core
    monkeypatch.setattr(chimerax.core, 'version', '1.5')
    texts = ['invalid requirement!', 'numpy', 'ChimeraX-Core>=1', 'other>=2']
    data = [{'bundle_name': name, 'bundle': {name: {'version': '1', 'requires': texts}}}
            for name in ('ChimeraX_First', 'ChimeraX_Second')]
    calls = []
    parse = available._parse_requires
    monkeypatch.setattr(available, '_parse_requires',
                        lambda text: calls.append(text) or parse(text))
    cache = available.AvailableBundleCache(None)
    cache._build_bundles(data)
    assert len(cache) == 2
    assert calls == texts[:3]
    first, second = cache
    first.requires[1].name = 'Changed'
    assert second.requires[1].name == 'ChimeraX-Core'
    # A caller's edits must take precedence over a cached decision.
    assert not cache._installable(first, Version('1.5'), {'ChimeraX-Core>=1': True})


def test_decisions_do_not_survive_catalog_loads(monkeypatch):
    import chimerax.core
    data = [{'bundle_name': 'ChimeraX_Example', 'bundle': {'ChimeraX_Example': {
        'version': '1', 'requires': ['ChimeraX-Core>=2']}}}]
    monkeypatch.setattr(chimerax.core, 'version', '1.5')
    old = available.AvailableBundleCache(None)
    old._build_bundles(data)
    assert not old and len(old.uninstallable) == 1
    monkeypatch.setattr(chimerax.core, 'version', '2.0')
    new = available.AvailableBundleCache(None)
    new._build_bundles(data)
    assert len(new) == 1 and not new.uninstallable


def cached_catalog(tmp_path, requires):
    import json
    data = [{'bundle_name': 'ChimeraX_Example', 'bundle': {'ChimeraX_Example': {
        'version': '1', 'requires': requires}}}]
    (tmp_path / 'available.json').write_text(json.dumps(data))


def test_disk_compatibility_cache_avoids_reparse_but_keeps_full_requirements(tmp_path, monkeypatch):
    import chimerax.core
    monkeypatch.setattr(chimerax.core, 'version', '1.5')
    texts = ['numpy', 'ChimeraX-Core>=1', 'other>=2']
    cached_catalog(tmp_path, texts)
    first = available.AvailableBundleCache(str(tmp_path))
    first.load_from_cache()
    calls = []
    parse = available._parse_requires
    monkeypatch.setattr(available, '_parse_requires', lambda text: calls.append(text) or parse(text))
    second = available.AvailableBundleCache(str(tmp_path))
    second.load_from_cache()
    assert calls == []
    assert [b.cache_data() for b in second] == [b.cache_data() for b in first]
    assert [str(r) for r in second[0].requires] == texts
    assert calls == texts


@pytest.mark.parametrize('change', ['core', 'parser', 'catalog'])
def test_disk_compatibility_cache_invalidates(tmp_path, monkeypatch, change):
    import chimerax.core
    import packaging
    monkeypatch.setattr(chimerax.core, 'version', '1.5')
    cached_catalog(tmp_path, ['ChimeraX-Core>=2'])
    first = available.AvailableBundleCache(str(tmp_path))
    first.load_from_cache()
    assert not first and len(first.uninstallable) == 1
    if change == 'core':
        monkeypatch.setattr(chimerax.core, 'version', '2.0')
    elif change == 'parser':
        monkeypatch.setattr(packaging, '__version__', packaging.__version__ + '.changed')
    else:
        cached_catalog(tmp_path, ['ChimeraX-Core>=1.000'])
    calls = []
    parse = available._parse_requires
    monkeypatch.setattr(available, '_parse_requires', lambda text: calls.append(text) or parse(text))
    second = available.AvailableBundleCache(str(tmp_path))
    second.load_from_cache()
    assert calls
    assert bool(second) == (change != 'parser')


def test_environment_markers_are_reevaluated_between_loads(tmp_path, monkeypatch):
    import chimerax.core
    from packaging.markers import Marker
    monkeypatch.setattr(chimerax.core, 'version', '1.5')
    cached_catalog(tmp_path, ['numpy', 'ChimeraX-Core>=1; python_version >= "3"'])
    monkeypatch.setattr(Marker, 'evaluate', lambda *a, **kw: True)
    first = available.AvailableBundleCache(str(tmp_path))
    first.load_from_cache()
    assert len(first) == 1
    monkeypatch.setattr(Marker, 'evaluate', lambda *a, **kw: False)
    second = available.AvailableBundleCache(str(tmp_path))
    second.load_from_cache()
    assert not second and len(second.uninstallable) == 1


@pytest.mark.parametrize('contents', ['not json', '{}', '[1,2]', '[[],{"x":1}]'])
def test_malformed_compatibility_cache_falls_back(tmp_path, monkeypatch, contents):
    import chimerax.core
    monkeypatch.setattr(chimerax.core, 'version', '1.5')
    cached_catalog(tmp_path, ['ChimeraX-Core>=1'])
    (tmp_path / 'core-compatibility.json').write_text(contents)
    cache = available.AvailableBundleCache(str(tmp_path))
    cache.load_from_cache()
    assert len(cache) == 1


def test_unwritable_compatibility_cache_does_not_prevent_loading(tmp_path, monkeypatch):
    import chimerax.core
    from chimerax.core import safesave
    monkeypatch.setattr(chimerax.core, 'version', '1.5')
    cached_catalog(tmp_path, ['ChimeraX-Core>=1'])
    def unwritable(*a, **kw):
        raise PermissionError('read-only cache directory')
    monkeypatch.setattr(safesave, 'SaveTextFile', unwritable)
    cache = available.AvailableBundleCache(str(tmp_path))
    cache.load_from_cache()
    assert len(cache) == 1


@pytest.mark.parametrize('bad_checks', [
    {'ChimeraX-Core>=1': 1},
    {'ChimeraX-Core>=1': 'true'},
    {'ChimeraX-Core>=1; python_version > "1"': True},
    ['not', 'a', 'mapping'],
])
def test_matching_cache_key_does_not_bypass_payload_validation(tmp_path, monkeypatch, bad_checks):
    import json
    import chimerax.core
    monkeypatch.setattr(chimerax.core, 'version', '1.5')
    cached_catalog(tmp_path, ['ChimeraX-Core>=1'])
    first = available.AvailableBundleCache(str(tmp_path))
    first.load_from_cache()
    path = tmp_path / 'core-compatibility.json'
    key, checks = json.loads(path.read_text())
    path.write_text(json.dumps([key, bad_checks]))
    calls = []
    parse = available._parse_requires
    monkeypatch.setattr(available, '_parse_requires', lambda text: calls.append(text) or parse(text))
    second = available.AvailableBundleCache(str(tmp_path))
    second.load_from_cache()
    assert len(second) == 1
    assert calls == ['ChimeraX-Core>=1']
