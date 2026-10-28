import os
import sys
from importlib.util import cache_from_source

import pytest

from chimerax.core import startup_prefetch


def touch(path, size=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x' * size)
    return str(path)


@pytest.fixture
def no_watch(monkeypatch):
    monkeypatch.setattr(startup_prefetch, '_opened', None)


def test_files_are_listed_in_startup_order_without_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(startup_prefetch, '_SYSTEM_PREFIXES', (str(tmp_path / 'system'),))
    lib = touch(tmp_path / 'lib' / 'libpython.dylib')
    system = touch(tmp_path / 'system' / 'libSystem.dylib')
    cocoa = touch(tmp_path / 'plugins' / 'platforms' / 'libqcocoa.dylib')
    offscreen = touch(tmp_path / 'plugins' / 'platforms' / 'libqoffscreen.dylib')
    gif = touch(tmp_path / 'plugins' / 'imageformats' / 'libqgif.dylib')
    site_dir = tmp_path / 'site-packages'
    pth = touch(site_dir / 'a.pth')
    touch(site_dir / 'not-a-pth.txt')
    source = touch(tmp_path / 'pkg' / 'chimerax_ui' / '__init__.py')
    compiled = touch(tmp_path / 'pkg' / 'chimerax_ui' / cache_from_source(source))
    icon = touch(tmp_path / 'pkg' / 'chimerax_ui' / 'icons' / 'a.png')
    extension = touch(tmp_path / 'pkg' / 'ext.so')
    settings = touch(tmp_path / 'config' / 'ui-1')
    files = startup_prefetch.startup_files(
        [('chimerax.ui', source), ('ext', extension), ('builtin', None), ('again', extension),
         ('missing', str(tmp_path / 'missing.py'))],
        images=[lib, system, cocoa, lib], opened=[settings, os.fsencode(settings), 'relative'],
        search_path=[str(site_dir), str(tmp_path / 'no-such-dir')])
    assert files == [lib, cocoa, pth, offscreen, gif, compiled, extension, settings, icon]


def test_large_data_files_are_not_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(startup_prefetch, '_MAX_DATA_FILE_SIZE', 10)
    small = touch(tmp_path / 'small', 10)
    large = touch(tmp_path / 'large', 11)
    files = startup_prefetch.startup_files([], images=[], opened=[small, large], search_path=[])
    assert files == [small]


def test_only_chimerax_packages_contribute_icons(tmp_path):
    other = touch(tmp_path / 'other' / '__init__.py')
    touch(tmp_path / 'other' / 'icons' / 'a.png')
    files = startup_prefetch.startup_files([('other', other)], images=[], search_path=[])
    assert not any(f.endswith('.png') for f in files)


def test_loaded_images_include_this_interpreter():
    if not sys.platform.startswith(('darwin', 'linux')):
        pytest.skip('Only macOS and Linux list loaded images')
    images = startup_prefetch.loaded_images()
    assert images
    assert any('python' in os.path.basename(p).lower() for p in images)


def test_nothing_is_recorded_without_the_launcher(monkeypatch, no_watch):
    monkeypatch.delenv(startup_prefetch.ENV_VAR, raising=False)
    startup_prefetch.watch_opened_files()
    assert startup_prefetch._opened is None
    assert startup_prefetch.record_startup_files() is None


def test_opened_files_are_collected_until_recorded(tmp_path, monkeypatch, no_watch):
    target = tmp_path / 'cache' / 'startup-prefetch-test'
    monkeypatch.setenv(startup_prefetch.ENV_VAR, str(target))
    data = touch(tmp_path / 'settings')
    later = touch(tmp_path / 'later')
    startup_prefetch.watch_opened_files()
    with open(data):
        pass
    thread = startup_prefetch.record_startup_files()
    thread.join(10)
    with open(later):
        pass
    listed = target.read_text().splitlines()
    assert data in listed
    assert later not in listed
    assert startup_prefetch._opened is None


def test_list_is_replaced_whole_and_only_when_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'path', [])
    target = tmp_path / 'cache' / 'list'
    first = touch(tmp_path / 'first.so')
    second = touch(tmp_path / 'second.so')
    monkeypatch.setattr(startup_prefetch, 'loaded_images', lambda: [first])
    startup_prefetch._write_list(str(target), [])
    assert target.read_text() == first + '\n'
    os.utime(target, (1, 1))
    startup_prefetch._write_list(str(target), [])
    assert target.stat().st_mtime == 1
    monkeypatch.setattr(startup_prefetch, 'loaded_images', lambda: [first, second])
    startup_prefetch._write_list(str(target), [])
    assert target.read_text() == first + '\n' + second + '\n'
    assert [p.name for p in target.parent.iterdir()] == ['list']


def test_write_failures_are_ignored(tmp_path, monkeypatch):
    blocker = touch(tmp_path / 'file')
    monkeypatch.setattr(startup_prefetch, 'loaded_images', lambda: [])
    startup_prefetch._write_list(os.path.join(blocker, 'list'), [])


def test_page_spans_join_nearby_pages():
    assert startup_prefetch._page_spans([], 4096) == []
    assert startup_prefetch._page_spans([3, 1, 2], 4096, gap=1) == [(4096, 3 * 4096)]
    # Pages 10 and 12 are one page apart, closer than the gap.
    assert startup_prefetch._page_spans([10, 12, 40], 4096, gap=8192) == [
        (10 * 4096, 3 * 4096), (40 * 4096, 4096)]
    assert startup_prefetch._page_spans([10, 12], 4096, gap=4096) == [
        (10 * 4096, 4096), (12 * 4096, 4096)]


def test_list_gives_ranges_after_a_tab():
    text = startup_prefetch._list_text(
        ['/lib/libQt6Core.so.6', '/pkg/__pycache__/a.pyc'],
        {'/lib/libQt6Core.so.6': [(0, 8192), (65536, 4096)]})
    assert text == '/lib/libQt6Core.so.6\t0+8192,65536+4096\n/pkg/__pycache__/a.pyc\n'


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='uses /proc/self/pagemap')
def test_mapped_ranges_cover_part_of_this_interpreters_libraries():
    import mmap
    ranges = startup_prefetch.mapped_ranges()
    assert ranges
    page = mmap.PAGESIZE
    for name, spans in ranges.items():
        size = os.path.getsize(name)
        assert spans == sorted(spans)
        for offset, length in spans:
            assert offset % page == 0 and length % page == 0 and length > 0
            assert offset < size
