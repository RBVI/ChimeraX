"""Catalog UI records remain deferred, mutable, and cache-compatible."""
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import sys
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest

from chimerax.core.toolshed import Toolshed, available, info as metadata


def catalog(**sections):
    return {
        'bundle_name': 'ChimeraX_Example',
        'bundle': {'ChimeraX_Example': {'version': '1', 'requires': ['ChimeraX-Core>=1']}},
        **sections,
    }


def test_build_and_provider_registration_do_not_construct_ui_records(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Constructed a catalog UI record during provider registration')
    for name in ('ToolInfo', 'CommandInfo', 'SelectorInfo'):
        monkeypatch.setattr(metadata, name, unexpected)
    bundle = available._build_bundle(catalog(
        tool={'Example Tool': {}}, command={'example': {}}, selector={'example': {}},
        manager={'open': {'autostart': 'true'}}, provider={'open/example': {'format': 'Example'}}))
    assert bundle.managers == {'open': {'autostart': 'true'}}
    assert list(bundle._providers_for_manager('open')) == [('example', {'format': 'Example'})]
    calls = []
    manager = SimpleNamespace(name='open',
        add_provider=lambda bi, name, **kw: calls.append((bi, name, kw)),
        end_providers=lambda: calls.append('done'))
    shed = SimpleNamespace(_installed_bundle_info=[], _available_bundle_info=[bundle],
                           _manager_instances={})
    Toolshed._init_single_manager(shed, manager)
    assert calls == [(bundle, 'example', {'format': 'Example'}), 'done']
    assert not {'tools', 'commands', 'selectors'} & vars(bundle).keys()


@pytest.mark.parametrize('section,attribute,record_class', [
    ('tool', 'tools', metadata.ToolInfo),
    ('command', 'commands', metadata.CommandInfo),
])
def test_field_defaults_order_and_exact_record_types(section, attribute, record_class):
    bundle = available._build_bundle(catalog(**{section: {
        'z-first': {'categories': ['One', 'Two'], 'synopsis': 'First synopsis'},
        'a-second': {},
        'empty-synopsis': {'synopsis': ''},
    }}))
    records = getattr(bundle, attribute)
    assert [record.name for record in records] == ['z-first', 'a-second', 'empty-synopsis']
    assert all(type(record) is record_class for record in records)
    assert records[0].categories == ['One', 'Two']
    assert records[0].synopsis == 'First synopsis'
    assert records[1].categories == []
    assert records[1].synopsis == records[2].synopsis == 'No synopsis given'


def test_selector_defaults_and_atomic_flag_semantics():
    bundle = available._build_bundle(catalog(selector={
        'missing-flag': {},
        'lower-false': {'atomic': 'false', 'synopsis': 'Non-atomic selector'},
        'upper-false': {'atomic': 'FALSE'},
        'mixed-false': {'atomic': 'False'},
        'true': {'atomic': 'true'},
        'empty': {'atomic': ''},
        'other-text': {'atomic': '0'},
    }))
    assert [record.name for record in bundle.selectors] == [
        'missing-flag', 'lower-false', 'upper-false', 'mixed-false', 'true', 'empty', 'other-text']
    assert all(type(record) is metadata.SelectorInfo for record in bundle.selectors)
    assert [record.atomic for record in bundle.selectors] == [True, False, False, False, True, True, True]
    assert bundle.selectors[0].synopsis == 'No synopsis given'
    assert bundle.selectors[1].synopsis == 'Non-atomic selector'


@pytest.mark.parametrize('section,attribute,class_name', [
    ('tool', 'tools', 'ToolInfo'),
    ('command', 'commands', 'CommandInfo'),
    ('selector', 'selectors', 'SelectorInfo'),
])
def test_first_access_constructs_only_requested_section_once(monkeypatch, section, attribute, class_name):
    bundle = available._build_bundle(catalog(tool={'tool': {}}, command={'command': {}},
                                            selector={'selector': {}}))
    calls = []
    for name in ('ToolInfo', 'CommandInfo', 'SelectorInfo'):
        original = getattr(metadata, name)
        def create(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(metadata, name, create)
    records = getattr(bundle, attribute)
    assert getattr(bundle, attribute) is records
    assert calls == [class_name]
    assert {name for name in ('tools', 'commands', 'selectors') if name in vars(bundle)} == {attribute}


@pytest.mark.parametrize('section,attribute', [('tool', 'tools'), ('command', 'commands'),
                                             ('selector', 'selectors')])
def test_cached_lists_support_mutation_and_replacement(section, attribute):
    bundle = available._build_bundle(catalog(**{section: {'first': {}, 'second': {}}}))
    records = getattr(bundle, attribute)
    records.reverse()
    records[0].synopsis = 'Edited by caller'
    assert getattr(bundle, attribute) is records
    assert [record.name for record in getattr(bundle, attribute)] == ['second', 'first']
    assert getattr(bundle, attribute)[0].synopsis == 'Edited by caller'
    replacement = [records[1]]
    setattr(bundle, attribute, replacement)
    assert getattr(bundle, attribute) is replacement
    assert bundle.cache_data()[2][attribute] == [records[1].cache_data()]


def test_replacement_before_first_access_does_not_construct_discarded_records(monkeypatch):
    bundle = available._build_bundle(catalog(tool={'tool': {}}, command={'command': {}},
                                            selector={'selector': {}}))
    def unexpected(*args, **kwargs):
        pytest.fail('Constructed records after caller replaced the deferred list')
    for name in ('ToolInfo', 'CommandInfo', 'SelectorInfo'):
        monkeypatch.setattr(metadata, name, unexpected)
    for attribute in ('tools', 'commands', 'selectors'):
        replacement = []
        setattr(bundle, attribute, replacement)
        assert getattr(bundle, attribute) is replacement
    assert all(bundle.cache_data()[2][attribute] == [] for attribute in ('tools', 'commands', 'selectors'))


@pytest.mark.parametrize('sections', [{}, {'tool': {}, 'command': {}, 'selector': {}}])
def test_missing_and_empty_sections_produce_independent_cached_lists(sections):
    first = available._build_bundle(catalog(**deepcopy(sections)))
    second = available._build_bundle(catalog(**deepcopy(sections)))
    for attribute in ('tools', 'commands', 'selectors'):
        records = getattr(first, attribute)
        assert records == []
        assert getattr(first, attribute) is records
        assert getattr(second, attribute) is not records


@pytest.mark.parametrize('section,attribute', [('tool', 'tools'), ('command', 'commands')])
def test_default_categories_are_independent_per_record_and_bundle(section, attribute):
    raw = catalog(**{section: {'first': {}, 'second': {}}})
    first, second = available._build_bundle(raw), available._build_bundle(raw)
    getattr(first, attribute)[0].categories.append('New category')
    assert getattr(first, attribute)[1].categories == []
    assert all(record.categories == [] for record in getattr(second, attribute))


@pytest.mark.parametrize('json_roundtrip', [False, True])
def test_cache_serialization_and_restoration_keep_records_and_metadata(json_roundtrip):
    bundle = available._build_bundle(catalog(
        tool={'Example Tool': {'categories': ['General'], 'synopsis': 'Tool synopsis'}},
        command={'example': {'categories': ['General']}},
        selector={'example': {'atomic': 'false'}},
        manager={'open': {'autostart': 'true'}}, provider={'open/example': {'format': 'Example'}}))
    assert not {'tools', 'commands', 'selectors'} & vars(bundle).keys()
    serialized = bundle.cache_data()
    assert {'tools', 'commands', 'selectors'} <= vars(bundle).keys()
    data = json.loads(json.dumps(serialized)) if json_roundtrip else deepcopy(serialized)
    for restore_class in (metadata.BundleInfo, type(bundle)):
        restored = restore_class.from_cache_data(deepcopy(data))
        assert type(restored.tools[0]) is metadata.ToolInfo
        assert type(restored.commands[0]) is metadata.CommandInfo
        assert type(restored.selectors[0]) is metadata.SelectorInfo
        assert restored.providers == bundle.providers
        assert restored.managers == bundle.managers
        assert restored.tools[0].name == 'Example Tool'
        assert restored.commands[0].synopsis == 'No synopsis given'
        assert restored.selectors[0].atomic is False
        if json_roundtrip:
            assert json.loads(json.dumps(restored.cache_data())) == data
        else:
            assert restored.cache_data() == serialized


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason='cached_property serializes first readers before Python 3.12')
@pytest.mark.parametrize('section,attribute,class_name', [
    ('tool', 'tools', 'ToolInfo'),
    ('command', 'commands', 'CommandInfo'),
    ('selector', 'selectors', 'SelectorInfo'),
])
def test_simultaneous_first_readers_share_one_published_list(monkeypatch, section, attribute, class_name):
    bundle = available._build_bundle(catalog(**{section: {'example': {}}}))
    original = getattr(metadata, class_name)
    barrier = Barrier(2, timeout=5)
    creations = []
    def create(*args, **kwargs):
        barrier.wait()
        record = original(*args, **kwargs)
        creations.append(record)
        return record
    monkeypatch.setattr(metadata, class_name, create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(getattr, bundle, attribute) for _ in range(2)]
        first, second = [future.result(timeout=5) for future in futures]
    assert len(creations) == 2
    assert first is second is getattr(bundle, attribute)
    assert len(first) == 1
    assert type(first[0]) is original


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason='cached_property serializes first readers before Python 3.12')
@pytest.mark.parametrize('section,attribute,class_name', [
    ('tool', 'tools', 'ToolInfo'),
    ('command', 'commands', 'CommandInfo'),
    ('selector', 'selectors', 'SelectorInfo'),
])
@pytest.mark.parametrize('edit', ['mutate', 'replace'])
def test_late_first_reader_honors_caller_edits_to_published_list(monkeypatch, section, attribute,
                                                               class_name, edit):
    bundle = available._build_bundle(catalog(**{section: {'example': {}}}))
    original = getattr(metadata, class_name)
    barrier, resume = Barrier(2, timeout=5), Event()
    counter_lock = Lock()
    entries = []
    def create(*args, **kwargs):
        with counter_lock:
            index = len(entries)
            entries.append(index)
        barrier.wait()
        if index == 1:
            assert resume.wait(timeout=5), 'Caller did not release the late reader'
        return original(*args, **kwargs)
    monkeypatch.setattr(metadata, class_name, create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(getattr, bundle, attribute) for _ in range(2)]
        try:
            done, pending = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            assert len(done) == len(pending) == 1
            published = done.pop().result()
            sentinel = object()
            if edit == 'mutate':
                published[0].synopsis = 'Caller edit'
                published.append(sentinel)
                expected = published
            else:
                expected = [sentinel]
                setattr(bundle, attribute, expected)
            resume.set()
            assert pending.pop().result(timeout=5) is expected
            assert getattr(bundle, attribute) is expected
            assert expected[-1] is sentinel
            if edit == 'mutate':
                assert expected[0].synopsis == 'Caller edit'
        finally:
            resume.set()
