"""Lazy built-in colors preserve their shared catalog and ordinary Color API."""
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_first_color_access_publishes_one_catalog_across_threads(monkeypatch):
    from chimerax.core import colors
    monkeypatch.delitem(vars(colors), 'BuiltinColors', raising=False)
    original = colors._builtin_colors
    calls = []
    def build():
        calls.append(True)
        return original()
    monkeypatch.setattr(colors, '_builtin_colors', build)
    assert 'BuiltinColors' in dir(colors) and 'BuiltinColors' in colors.__all__
    assert 'BuiltinColors' not in vars(colors)
    with ThreadPoolExecutor(max_workers=4) as pool:
        catalogs = list(pool.map(lambda _: colors.BuiltinColors, range(4)))
    assert all(c is catalogs[0] for c in catalogs)
    assert calls == [True]
    assert colors.UserColors()['red'] is catalogs[0]['red']
    red = colors.Color('red')
    assert red.rgba is not catalogs[0]['red'].rgba
    assert tuple(red.uint8x4()) == (255, 0, 0, 255)


def test_failed_catalog_creation_can_retry_without_publishing_partial_values(monkeypatch):
    from chimerax.core import colors
    monkeypatch.delitem(vars(colors), 'BuiltinColors', raising=False)
    original = colors._builtin_colors
    calls = []
    def build():
        calls.append(True)
        if len(calls) == 1:
            raise ValueError('unavailable')
        return original()
    monkeypatch.setattr(colors, '_builtin_colors', build)
    with pytest.raises(ValueError):
        colors.BuiltinColors
    assert 'BuiltinColors' not in vars(colors)
    assert colors.BuiltinColors['red']
    assert len(calls) == 2


def test_wildcard_import_and_explicit_catalog_assignment_keep_identity(monkeypatch):
    from chimerax.core import colors
    custom = {'red': colors.Color((0, 1, 0, 1))}
    monkeypatch.setitem(vars(colors), 'BuiltinColors', custom)
    namespace = {}
    exec('from chimerax.core.colors import *', namespace)
    assert namespace['BuiltinColors'] is custom
    assert tuple(colors.Color('red').uint8x4()) == (0, 255, 0, 255)
