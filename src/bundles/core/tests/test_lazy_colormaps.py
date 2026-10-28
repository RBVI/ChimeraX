from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest


@pytest.fixture
def unloaded_colormaps(monkeypatch):
    from chimerax.core import colors
    # Restore even an initially absent module attribute after lazy lookup.
    monkeypatch.setitem(colors.__dict__, 'BuiltinColormaps', object())
    monkeypatch.delitem(colors.__dict__, 'BuiltinColormaps')
    return colors


def test_first_lookup_publishes_one_shared_dictionary(unloaded_colormaps, monkeypatch):
    colors = unloaded_colormaps
    from sortedcontainers import SortedDict
    palette = object()
    catalog = SortedDict({'example': palette, 'alias': palette})
    calls = []
    monkeypatch.setattr(colors, '_builtin_colormaps', lambda: calls.append(1) or catalog)
    assert 'BuiltinColormaps' in dir(colors)
    assert 'BuiltinColormaps' in colors.__all__
    assert calls == []
    assert colors.BuiltinColormaps is catalog
    assert colors.BuiltinColormaps is catalog
    assert list(catalog) == ['alias', 'example']
    assert catalog['alias'] is catalog['example']
    catalog['added'] = palette
    assert colors.BuiltinColormaps['added'] is palette
    assert calls == [1]


def test_internal_helpers_initialize_catalog(unloaded_colormaps, monkeypatch):
    colors = unloaded_colormaps
    palette = colors.Colormap((0, 1), ((0, 0, 0, 1), (1, 1, 1, 1)))
    monkeypatch.setattr(colors, '_builtin_colormaps', lambda: {'example': palette})
    assert colors.colormap_with_range(None, None, 'example') is palette
    assert colors.palette_name(palette.colors) == 'example'


def test_failed_initialization_can_be_retried(unloaded_colormaps, monkeypatch):
    colors = unloaded_colormaps
    def fail():
        raise ValueError('invalid palette data')
    monkeypatch.setattr(colors, '_builtin_colormaps', fail)
    with pytest.raises(ValueError, match='invalid palette data'):
        colors.BuiltinColormaps
    assert 'BuiltinColormaps' not in vars(colors)
    catalog = {}
    monkeypatch.setattr(colors, '_builtin_colormaps', lambda: catalog)
    assert colors.BuiltinColormaps is catalog


def test_concurrent_readers_share_the_same_catalog(unloaded_colormaps, monkeypatch):
    colors = unloaded_colormaps
    entered, release = Event(), Event()
    calls = []
    catalog = {}
    def build():
        calls.append(1)
        entered.set()
        assert release.wait(5), 'catalog initialization was not released'
        return catalog
    monkeypatch.setattr(colors, '_builtin_colormaps', build)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(lambda: colors.BuiltinColormaps)
        try:
            assert entered.wait(5), 'catalog initialization did not start'
            second = workers.submit(lambda: colors.BuiltinColormaps)
        finally:
            release.set()
        assert first.result(timeout=5) is catalog
        assert second.result(timeout=5) is catalog
    assert calls == [1]
