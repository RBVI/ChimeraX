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


