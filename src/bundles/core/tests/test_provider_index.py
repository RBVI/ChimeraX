"""Provider lookup preserves registration order and mutable bundle metadata."""
from types import SimpleNamespace

from chimerax.core.toolshed import Toolshed
from chimerax.core.toolshed.info import BundleInfo


def bundle(name, providers, installed=True):
    return BundleInfo(name, installed, providers=providers)


def test_provider_names_group_by_first_slash_in_original_order():
    info = bundle('example', {
        'open/first': {'format': 'one'},
        'save/first': {},
        'open/database/second': {'format': 'two'},
    })
    assert list(info._providers_for_manager('open')) == [
        ('first', {'format': 'one'}), ('database/second', {'format': 'two'})]
    index = info._provider_index
    assert list(info._providers_for_manager('save')) == [('first', {})]
    assert list(info._providers_for_manager('missing')) == []
    assert info._provider_index is index


def test_provider_options_are_live_when_replaced_or_edited():
    info = bundle('example', {'open/file': {'enabled': True}})
    list(info._providers_for_manager('open'))
    index = info._provider_index
    info.providers['open/file']['enabled'] = False
    assert list(info._providers_for_manager('open')) == [('file', {'enabled': False})]
    info.providers = {'open/file': {'replacement': True}}
    assert list(info._providers_for_manager('open')) == [('file', {'replacement': True})]
    assert info._provider_index is index


def test_provider_name_add_remove_and_reorder_refresh_index():
    info = bundle('example', {'open/a': {}, 'open/b': {}})
    assert [n for n, kw in info._providers_for_manager('open')] == ['a', 'b']
    info.providers['open/c'] = {}
    del info.providers['open/b']
    assert [n for n, kw in info._providers_for_manager('open')] == ['a', 'c']
    info.providers['open/a'] = info.providers.pop('open/a')
    assert [n for n, kw in info._providers_for_manager('open')] == ['c', 'a']
    info.providers.clear()
    assert list(info._providers_for_manager('open')) == []


def test_provider_index_does_not_change_serialized_bundle_metadata():
    info = bundle('example', {'open/a': {'suffix': '.a'}})
    before = info.cache_data()
    list(info._providers_for_manager('open'))
    assert info.cache_data() == before
    restored = BundleInfo.from_cache_data(before)
    assert list(restored._providers_for_manager('open')) == [('a', {'suffix': '.a'})]


def test_manager_registration_preserves_bundle_order_and_finishes_once():
    installed = bundle('installed', {'open/a': {}, 'save/b': {}, 'open/c': {'x': 1}})
    available = bundle('available', {'open/a': {'x': 2}}, installed=False)
    calls = []
    manager = SimpleNamespace(name='open',
        add_provider=lambda bi, name, **kw: calls.append((bi.name, name, kw)),
        end_providers=lambda: calls.append('done'))
    ts = SimpleNamespace(_installed_bundle_info=[installed],
        _available_bundle_info=[available], _manager_instances={})
    Toolshed._init_single_manager(ts, manager)
    assert ts._manager_instances['open'] is manager
    assert calls == [('installed', 'a', {}), ('installed', 'c', {'x': 1}),
                     ('available', 'a', {'x': 2}), 'done']


def test_manager_without_available_bundles_still_finishes():
    calls = []
    manager = SimpleNamespace(name='empty',
        add_provider=lambda *a, **kw: calls.append('unexpected provider'),
        end_providers=lambda: calls.append('done'))
    ts = SimpleNamespace(_installed_bundle_info=[bundle('installed', {'open/a': {}})],
        _available_bundle_info=None, _manager_instances={})
    Toolshed._init_single_manager(ts, manager)
    assert calls == ['done']


def test_empty_bundles_skip_provider_iteration_but_later_additions_are_registered():
    class CheckedBundle(BundleInfo):
        def _providers_for_manager(self, manager_name):
            assert self.providers, 'Iterated an empty provider dictionary'
            yield from super()._providers_for_manager(manager_name)

    installed_empty = CheckedBundle('installed-empty', True)
    available_empty = CheckedBundle('available-empty', False)
    populated = bundle('populated', {'open/existing': {'x': 1}})
    calls = []
    manager = SimpleNamespace(name='open',
        add_provider=lambda bi, name, **kw: calls.append((bi.name, name, kw)),
        end_providers=lambda: calls.append('done'))
    ts = SimpleNamespace(_installed_bundle_info=[installed_empty, populated],
        _available_bundle_info=[available_empty], _manager_instances={})
    Toolshed._init_single_manager(ts, manager)
    assert calls == [('populated', 'existing', {'x': 1}), 'done']
    installed_empty.providers['open/new-installed'] = {'x': 2}
    available_empty.providers['open/new-available'] = {'x': 3}
    calls.clear()
    Toolshed._init_single_manager(ts, manager)
    assert calls == [('installed-empty', 'new-installed', {'x': 2}),
                     ('populated', 'existing', {'x': 1}),
                     ('available-empty', 'new-available', {'x': 3}), 'done']
