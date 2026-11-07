"""prereqs/pips/patch_pyside6.py replaces the signature bootstrap embedded in
libshiboken.  Linux and macOS embed it with LF line endings, Windows with
CRLF; both must be found and overwritten in place."""

import importlib.util
import os

import pytest

_PATCH = os.path.join(os.path.dirname(__file__), "..", "prereqs", "pips", "patch_pyside6.py")


@pytest.fixture
def patch_pyside6(monkeypatch):
    spec = importlib.util.spec_from_file_location("patch_pyside6", _PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # replace_library signs the library on macOS; these are not real libraries.
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kw: None)
    return module


def _embedded_bootstrap(eol, size):
    lines = ["# Copyright (C) 2022 The Qt Company Ltd.", "",
             "def find_incarnated_files():", "    pass", "",
             "def bootstrap():", "    pass", ""]
    text = eol.join(lines).encode()
    return text + b"#" * (size - len(text))


@pytest.mark.parametrize("eol, name", [
    ("\n", "libshiboken6.abi3.so.6.11"),
    ("\r\n", "shiboken6.abi3.dll"),
])
def test_bootstrap_is_replaced_in_place(patch_pyside6, tmp_path, eol, name):
    replacement = open(os.path.join(os.path.dirname(_PATCH), "pyside6_signature_bootstrap.py"),
                       encoding="utf-8").read().encode()
    embedded = _embedded_bootstrap(eol, len(replacement) + 100)
    library = tmp_path / name
    library.write_bytes(b"\0code before\0" + embedded + b"\0code after\0")
    files_dir = tmp_path / "files.dir"
    files_dir.mkdir()

    patch_pyside6.patch_bootstrap(str(tmp_path), str(files_dir))

    data = library.read_bytes()
    assert len(data) == len(b"\0code before\0" + embedded + b"\0code after\0")
    start = len(b"\0code before\0")
    patched = data[start:start + len(embedded)]
    assert patched.rstrip(b" ") == replacement
    assert data.endswith(b"\0code after\0")
    # Written in text mode, so CRLF on Windows.
    assert (files_dir / "signature_bootstrap.py").read_text(encoding="utf-8") == replacement.decode()

    # Running it again leaves the library alone.
    patch_pyside6.patch_bootstrap(str(tmp_path), str(files_dir))
    assert library.read_bytes() == data


def test_missing_bootstrap_stops_the_build(patch_pyside6, tmp_path):
    (tmp_path / "shiboken6.abi3.dll").write_bytes(b"\0no bootstrap here\0")
    files_dir = tmp_path / "files.dir"
    files_dir.mkdir()
    with pytest.raises(SystemExit, match="embedded signature_bootstrap not found"):
        patch_pyside6.patch_bootstrap(str(tmp_path), str(files_dir))
