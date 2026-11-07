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
    monkeypatch.setattr(startup_prefetch, '_noted', None)


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


@pytest.fixture
def mac_launchers(tmp_path, monkeypatch):
    contents = tmp_path / 'ChimeraX.app' / 'Contents'
    cli = contents / 'bin' / 'ChimeraX'
    dock = contents / 'MacOS' / 'ChimeraX'
    touch(cli)
    dock.parent.mkdir()
    os.link(cli, dock)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(sys, 'executable', str(cli))
    monkeypatch.setattr(os.path, 'expanduser', lambda path: str(tmp_path))
    cache = tmp_path / 'Library' / 'Caches' / 'ChimeraX'
    return cli, dock, cache


def test_both_mac_entry_points_record_lists_read_by_existing_launchers(mac_launchers, monkeypatch):
    cli, dock, cache = mac_launchers
    first = cache / startup_prefetch._launcher_list_name(cli)
    second = cache / startup_prefetch._launcher_list_name(dock)
    assert first != second
    assert startup_prefetch._list_destinations(str(first)) == [str(first), str(second)]
    monkeypatch.setattr(sys, 'executable', str(dock))
    assert startup_prefetch._list_destinations(str(second)) == [str(second), str(first)]


def test_existing_unchanged_list_does_not_prevent_repairing_missing_entry_point(mac_launchers, monkeypatch):
    cli, dock, cache = mac_launchers
    first = cache / startup_prefetch._launcher_list_name(cli)
    second = cache / startup_prefetch._launcher_list_name(dock)
    monkeypatch.setattr(startup_prefetch, 'startup_files', lambda *args, **kw: ['/library'])
    monkeypatch.setattr(startup_prefetch, 'mapped_ranges', lambda: {})
    first.parent.mkdir(parents=True)
    first.write_text('/library\n')
    os.utime(first, (1, 1))
    startup_prefetch._write_list(str(first), [])
    assert first.stat().st_mtime == 1
    assert second.read_text() == '/library\n'
    assert set(p.name for p in cache.iterdir()) == {first.name, second.name}


def test_custom_prefetch_destinations_do_not_touch_user_cache(mac_launchers, tmp_path):
    assert startup_prefetch._list_destinations(str(tmp_path / 'isolated/list')) == [
        str(tmp_path / 'isolated/list')]


@pytest.mark.parametrize('case', ['missing', 'different_file', 'linux'])
def test_alias_is_limited_to_same_mac_executable(mac_launchers, monkeypatch, case):
    cli, dock, cache = mac_launchers
    path = str(cache / startup_prefetch._launcher_list_name(cli))
    if case == 'missing':
        dock.unlink()
    elif case == 'different_file':
        dock.unlink()
        touch(dock)
    else:
        monkeypatch.setattr(sys, 'platform', 'linux')
    assert startup_prefetch._list_destinations(path) == [path]


def test_failure_updating_one_list_does_not_suppress_the_other(mac_launchers, monkeypatch):
    cli, dock, cache = mac_launchers
    first = str(cache / startup_prefetch._launcher_list_name(cli))
    second = str(cache / startup_prefetch._launcher_list_name(dock))
    calls = []
    def write(path, text):
        calls.append(path)
        if path == first:
            raise OSError('read only')
    monkeypatch.setattr(startup_prefetch, 'startup_files', lambda *args, **kw: [])
    monkeypatch.setattr(startup_prefetch, 'mapped_ranges', lambda: {})
    monkeypatch.setattr(startup_prefetch, '_write_list_text', write)
    startup_prefetch._write_list(first, [])
    assert calls == [first, second]


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


def test_windows_system_dlls_are_left_out(monkeypatch):
    monkeypatch.setattr(startup_prefetch, '_SYSTEM_PREFIXES', (os.path.normcase('/sysroot') + os.sep,))
    assert startup_prefetch._is_system('/sysroot/System32/kernel32.dll')
    assert not startup_prefetch._is_system('/app/bin/python314.dll')


@pytest.mark.skipif(sys.platform != 'win32', reason='uses QueryWorkingSetEx')
def test_windows_mapped_ranges_cover_part_of_the_python_dll():
    ranges = startup_prefetch.mapped_ranges()
    python_dll = [name for name in ranges if os.path.basename(name).lower().startswith('python3')]
    assert python_dll
    for name in python_dll:
        spans = ranges[name]
        # Offsets are in the loaded image, which may be larger than the file.
        assert spans == sorted(spans)
        assert all(offset % 4096 == 0 and length > 0 for offset, length in spans)


def test_list_is_written_with_lf_line_endings(tmp_path):
    path = tmp_path / 'list'
    startup_prefetch._write_list_text(str(path), 'a\nb\n')
    assert path.read_bytes() == b'a\nb\n'


def test_windows_dlls_are_listed_just_before_their_package(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(startup_prefetch, '_SYSTEM_PREFIXES', ())
    bin_dir = tmp_path / 'bin'
    site = bin_dir / 'Lib' / 'site-packages'
    python_dll = touch(bin_dir / 'python314.dll')
    vcruntime = touch(bin_dir / 'vcruntime140.dll')
    ctypes_pyd = touch(bin_dir / 'DLLs' / '_ctypes.pyd')
    libffi = touch(bin_dir / 'DLLs' / 'libffi-8.dll')
    openblas = touch(site / 'numpy.libs' / 'openblas.dll')
    numpy_init = touch(site / 'numpy' / '__init__.py')
    numpy_pyc = touch(type(site)(cache_from_source(numpy_init)))
    multiarray = touch(site / 'numpy' / '_core' / '_multiarray.pyd')
    gallium = touch(bin_dir / 'libgallium_wgl.dll')
    # EnumProcessModules order: Python's DLLs, then extension modules as they load.
    images = [python_dll, vcruntime, ctypes_pyd, libffi, openblas, multiarray, gallium]
    modules = [('_ctypes', ctypes_pyd), ('numpy', numpy_init), ('numpy._core._multiarray', multiarray)]
    search_path = [str(bin_dir / 'DLLs'), str(bin_dir / 'Lib'), str(bin_dir), str(site)]
    files = startup_prefetch.startup_files(modules, images=images, search_path=search_path)
    assert files == [python_dll, vcruntime, libffi, ctypes_pyd, openblas, numpy_pyc,
                     multiarray, gallium]


def test_package_key_uses_the_innermost_search_directory(tmp_path):
    lib = tmp_path / 'Lib'
    site = lib / 'site-packages'
    path = [str(lib), str(site)]
    key = startup_prefetch._package_key
    assert key(str(site / 'numpy.libs' / 'x.dll'), path) == key(str(site / 'numpy' / 'core.pyd'), path)
    assert key(str(site / 'PySide6' / 'Qt6Core.dll'), path) == (os.path.normcase(str(site)), 'PySide6')
    assert key(str(tmp_path / 'DLLs' / 'libffi.dll'), path) == key(str(tmp_path / 'DLLs' / '_ctypes.pyd'), path)


def test_module_sources_are_opened_but_not_read(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    source = touch(tmp_path / 'pkg' / 'mod.py')
    cached = touch(type(tmp_path)(cache_from_source(source)))
    library = touch(tmp_path / 'lib.so')
    files, sources = startup_prefetch._with_sources([library, cached])
    assert files == [library, source, cached]
    assert sources == {source}
    text = startup_prefetch._list_text(files, {library: [(0, 4096)]}, sources)
    assert text == f'{library}\t0+4096\n{source}\t\n{cached}\n'


@pytest.mark.parametrize('platform', ['darwin', 'linux'])
def test_only_windows_lists_sources(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, 'platform', platform)
    cached = touch(type(tmp_path)(cache_from_source(touch(tmp_path / 'mod.py'))))
    assert startup_prefetch._with_sources([cached]) == ([cached], set())


def test_noted_files_are_listed_until_recorded(tmp_path, monkeypatch, no_watch):
    target = tmp_path / 'cache' / 'startup-prefetch-noted'
    monkeypatch.setenv(startup_prefetch.ENV_VAR, str(target))
    icon = touch(tmp_path / 'icons' / 'tool.png')
    later = touch(tmp_path / 'icons' / 'later.png')
    startup_prefetch.note_file(icon)	# not watching yet: ignored
    startup_prefetch.watch_opened_files()
    startup_prefetch.note_file(icon)
    startup_prefetch.record_startup_files().join(10)
    startup_prefetch.note_file(later)	# after recording: ignored
    listed = [line.split('\t')[0] for line in target.read_text().splitlines()]
    assert icon in listed
    assert later not in listed


def test_bundle_paths_found_at_startup_are_noted(tmp_path, monkeypatch, no_watch):
    from chimerax.core.toolshed.info import BundleInfo
    icon = touch(tmp_path / 'pkg' / 'icons' / 'tool.png')
    bundle = BundleInfo('ChimeraX-Test', installed=True, packages=[(('chimerax', 'test'), str(tmp_path / 'pkg'))])
    monkeypatch.setattr(bundle, '_package_locations', lambda: (str(tmp_path / 'pkg'),))
    monkeypatch.setattr(startup_prefetch, '_noted', [])
    assert bundle.get_path('icons/tool.png') == icon
    assert bundle.get_path('icons/missing.png') is None
    assert startup_prefetch._noted == [icon]


def test_package_locations_use_the_imported_parent_path(tmp_path, monkeypatch):
    import types
    from chimerax.core.toolshed.info import BundleInfo
    parent = types.ModuleType('cxtestparent')
    parent.__path__ = [str(tmp_path / 'a'), str(tmp_path / 'b')]
    (tmp_path / 'b' / 'child').mkdir(parents=True)
    monkeypatch.setitem(sys.modules, 'cxtestparent', parent)
    import importlib.util
    def no_find_spec(name):
        raise AssertionError('find_spec should not be needed')
    monkeypatch.setattr(importlib.util, 'find_spec', no_find_spec)
    bundle = BundleInfo('ChimeraX-Test', installed=True, api_package_name='cxtestparent.child')
    assert bundle._package_locations() == (str(tmp_path / 'b' / 'child'),)


def test_windows_lists_noted_files_for_lookup_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'path', [])
    monkeypatch.setattr(startup_prefetch, '_SYSTEM_PREFIXES', ())
    monkeypatch.setattr(startup_prefetch, 'mapped_ranges', lambda: {})
    library = touch(tmp_path / 'lib.dll')
    icon = touch(tmp_path / 'icons' / 'tool.png')
    monkeypatch.setattr(startup_prefetch, 'loaded_images', lambda: [library])
    target = tmp_path / 'list'
    startup_prefetch._write_list(str(target), [], opened=[icon], noted=[icon, icon])
    assert target.read_text() == f'{library}\n{icon}\t\n'
