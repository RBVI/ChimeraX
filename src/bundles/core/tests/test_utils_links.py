"""Link helpers load Windows process support only on the Windows path."""
import builtins
import sys
from types import SimpleNamespace


def test_posix_link_preserves_target_and_does_not_import_subprocess(monkeypatch):
    from chimerax.core import utils
    calls = []
    original_import = builtins.__import__
    def import_module(name, *args, **kwargs):
        if name == 'subprocess':
            raise AssertionError('POSIX link creation imported Windows process support')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(utils, 'sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(utils.os, 'symlink', lambda target, source: calls.append((target, source)))
    monkeypatch.setattr(builtins, '__import__', import_module)
    utils.make_link('/target with spaces', '/link with spaces')
    assert calls == [('/target with spaces', '/link with spaces')]


def test_windows_link_imports_process_support_when_used(monkeypatch):
    from chimerax.core import utils
    calls = []
    monkeypatch.setattr(utils, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setitem(sys.modules, 'subprocess', SimpleNamespace(run=calls.append))
    utils.make_link('C:/target with spaces', 'C:/junction with spaces')
    assert calls == [['mklink', '/J', 'C:/junction with spaces', 'C:/target with spaces']]
