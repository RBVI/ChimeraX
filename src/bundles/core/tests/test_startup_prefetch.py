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
    if sys.platform not in ('darwin', 'linux', 'win32'):
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
    monkeypatch.setattr(startup_prefetch, '_write_list_text', write)
    startup_prefetch._write_list(first, [])
    assert calls == [first, second]


def test_linux_loaded_images_ignore_anonymous_mappings(monkeypatch):
    from io import StringIO
    from unittest.mock import mock_open, patch

    monkeypatch.setattr(sys, 'platform', 'linux')
    maps = StringIO(
        '1000-2000 r-xp 00000000 08:01 1 /lib/library with spaces.so\n'
        '2000-3000 rw-p 00000000 00:00 0 [heap]\n'
    )
    with patch.object(startup_prefetch, 'open', mock_open(read_data=maps.read()), create=True):
        assert startup_prefetch.loaded_images() == ['/lib/library with spaces.so']



def test_windows_system_dlls_are_left_out(monkeypatch):
    monkeypatch.setattr(startup_prefetch, '_SYSTEM_PREFIXES', (os.path.normcase('/sysroot') + os.sep,))
    assert startup_prefetch._is_system('/sysroot/System32/kernel32.dll')
    assert not startup_prefetch._is_system('/app/bin/python314.dll')



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
