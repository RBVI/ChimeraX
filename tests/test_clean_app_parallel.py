"""clean_app.py runs its removals at once where they touch different parts of
the application, and prunes PySide6 at the end of the build."""

from pathlib import Path
import importlib.util
import subprocess
import threading
import time

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_app():
    return _load("clean_app", _ROOT / "utils/clean_app.py")


@pytest.fixture
def prune_pyside6():
    return _load("prune_pyside6", _ROOT / "prereqs/pips/prune_pyside6.py")


def test_thin_macos_binaries_thins_each_universal_binary(clean_app, tmp_path,
                                                         monkeypatch):
    fat = b"\xca\xfe\xba\xbe" + bytes(96)
    paths = []
    for i in range(5):
        path = tmp_path / f"lib{i}.dylib"
        path.write_bytes(fat)
        paths.append(path)
    (tmp_path / "thin.dylib").write_bytes(b"\xcf\xfa\xed\xfe" + bytes(96))

    def run(cmd, **kw):
        if cmd[:2] == ["lipo", "-archs"]:
            return subprocess.CompletedProcess(cmd, 0, "x86_64 arm64\n", "")
        if cmd[:2] == ["lipo", "-thin"]:
            Path(cmd[cmd.index("-output") + 1]).write_bytes(b"thin")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(cmd)

    monkeypatch.setattr(clean_app.subprocess, "run", run)
    freed = clean_app.thin_macos_binaries(str(tmp_path), "arm64")
    assert freed == 5 * (len(fat) - len(b"thin"))
    assert all(path.read_bytes() == b"thin" for path in paths)
    assert not list(tmp_path.glob("*.thin"))


def test_check_libs_checks_every_binary(prune_pyside6, tmp_path, monkeypatch):
    macho = b"\xcf\xfa\xed\xfe" + bytes(12)
    for name in ("a.so", "b.so", "sub/c.so"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(macho)
    (tmp_path / "readme.txt").write_text("not a binary")
    monkeypatch.setattr(prune_pyside6, "linked_qt_libs",
                        lambda path: {"Core", "Pdf"} if path.endswith("c.so")
                        else {"Core"})
    assert prune_pyside6.check_libs(str(tmp_path), {"Pdf"}) == {
        str(Path("sub/c.so")): {"Pdf"}
    }
