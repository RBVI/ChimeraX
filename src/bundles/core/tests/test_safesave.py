"""Safe saves must use independent temporary files without process inspection."""

import builtins
import gzip
import importlib.util
from pathlib import Path

import pytest


def test_import_does_not_require_psutil(monkeypatch):
    original_import = builtins.__import__

    def no_process_inspection(name, *args, **kw):
        if name == 'psutil' or name.startswith('psutil.'):
            raise AssertionError('Saving a file must not import psutil')
        return original_import(name, *args, **kw)

    monkeypatch.setattr(builtins, '__import__', no_process_inspection)
    path = Path(__file__).parents[1] / 'src' / 'safesave.py'
    spec = importlib.util.spec_from_file_location('safesave_without_psutil', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_overlapping_saves_use_independent_files(tmp_path):
    from chimerax.core.safesave import SaveTextFile
    path = tmp_path / 'settings'
    path.write_text('original')
    with SaveTextFile(str(path)) as first:
        first.write('first')
        with SaveTextFile(str(path)) as second:
            second.write('second')
            assert first.name != second.name
            assert Path(first.name).parent == tmp_path
            assert Path(second.name).parent == tmp_path
            assert path.read_text() == 'original'
        assert path.read_text() == 'second'
    assert path.read_text() == 'first'
    assert list(tmp_path.iterdir()) == [path]


def test_failed_save_does_not_remove_another_writers_file(tmp_path):
    from chimerax.core.safesave import SaveTextFile
    path = tmp_path / 'settings'
    path.write_text('original')
    with SaveTextFile(str(path)) as first:
        first.write('first')
        with pytest.raises(RuntimeError, match='cancel'):
            with SaveTextFile(str(path)) as second:
                second.write('discard')
                raise RuntimeError('cancel')
        assert path.read_text() == 'original'
        assert Path(first.name).exists()
        assert not Path(second.name).exists()
    assert path.read_text() == 'first'
    assert list(tmp_path.iterdir()) == [path]


def test_custom_compressed_writer(tmp_path):
    from chimerax.core.safesave import SaveFile
    path = tmp_path / 'session.gz'
    payload = b'session data' * 100
    with SaveFile(str(path), open=lambda p: gzip.open(p, 'wb')) as output:
        output.write(payload)
    with gzip.open(path, 'rb') as saved:
        assert saved.read() == payload
    assert list(tmp_path.iterdir()) == [path]


def test_critical_save_flushes_before_replacement(tmp_path, monkeypatch):
    from chimerax.core import safesave
    path = tmp_path / 'settings'
    flushed = []
    monkeypatch.setattr(safesave.os, 'fsync', lambda fd: flushed.append(fd))
    with safesave.SaveTextFile(str(path), critical=True) as output:
        output.write('persist')
        descriptor = output.fileno()
    assert flushed == [descriptor]
    assert path.read_text() == 'persist'
