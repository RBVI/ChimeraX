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


def _record_steps(clean_app, monkeypatch, prune_ok=True):
    events = []
    lock = threading.Lock()

    def step(name, result=None, wait=0):
        def run(*args):
            with lock:
                events.append(("start", name))
            time.sleep(wait)
            with lock:
                events.append(("end", name))
            return result

        return run

    for name in ("remove_package_tests", "remove_tkinter",
                 "remove_babel_locales"):
        monkeypatch.setattr(clean_app, name, step(name, wait=0.05))
    monkeypatch.setattr(clean_app, "prune_pyside6",
                        step("prune_pyside6", prune_ok, wait=0.05))
    # strip_debug_info too: on Linux it really strips whatever is at the path,
    # and in the Flatpak build /app is the app running the tests
    for name in ("remove_pycache", "remove_foreign_binaries",
                 "thin_macos_binaries", "strip_debug_info",
                 "remove_windows_debug_symbols"):
        monkeypatch.setattr(clean_app, name, step(name))
    return events


def test_removals_run_at_once_then_the_whole_tree_steps(clean_app, monkeypatch,
                                                        tmp_path):
    events = _record_steps(clean_app, monkeypatch)
    assert clean_app.clean_app(str(tmp_path)) is True
    first = {"prune_pyside6", "remove_package_tests", "remove_tkinter",
             "remove_babel_locales"}
    # All of the first steps start before any of them ends
    first_end = min(i for i, (kind, name) in enumerate(events)
                    if kind == "end" and name in first)
    assert {name for kind, name in events[:first_end]} == first
    # remove_pycache starts after they have all ended
    pycache = events.index(("start", "remove_pycache"))
    assert {name for kind, name in events[:pycache] if kind == "end"} == first
    # and the binary steps after it
    later = [name for kind, name in events[pycache + 1:] if kind == "start"]
    assert "remove_foreign_binaries" in later


def test_failed_prune_fails_clean_app(clean_app, monkeypatch, tmp_path):
    _record_steps(clean_app, monkeypatch, prune_ok=False)
    assert clean_app.clean_app(str(tmp_path)) is False


@pytest.mark.parametrize("system,keep_debug_info,removed", [
    ("win32", False, True),
    ("win32", True, False),
    ("linux", False, False),
    ("darwin", False, False),
])
def test_windows_symbol_cleanup_respects_platform_and_debug_flag(
        clean_app, monkeypatch, tmp_path, system, keep_debug_info, removed):
    events = _record_steps(clean_app, monkeypatch)
    monkeypatch.setattr(clean_app.sysconfig, "get_path", lambda _: str(tmp_path))
    monkeypatch.setattr(clean_app.sys, "platform", system)
    assert clean_app.clean_app(str(tmp_path), keep_debug_info) is True
    assert (("start", "remove_windows_debug_symbols") in events) is removed


def test_remove_windows_debug_symbols_keeps_molecular_pdbs(clean_app, tmp_path):
    symbol_data = b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\x00\x00\x00" + bytes(64)
    symbols = []
    for name in ("bin/DLLs/_ssl.pdb", "bin/Lib/site-packages/debugpy/helper.PDB",
                 "bin/Lib/venv/scripts/nt/venvlauncher.pdb"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(symbol_data)
        symbols.append(path)
    keep = {
        "molecule.pdb": b"HEADER    PROTEIN\nATOM      1  N   ALA A   1\n",
        "MODEL.PDB": b"MODEL        1\nENDMDL\n",
        "empty.pdb": b"",
        "text.pdb": b"Microsoft C/C++ documentation\n",
        "other.dat": symbol_data,
    }
    for name, data in keep.items():
        (tmp_path / name).write_bytes(data)

    assert clean_app.remove_windows_debug_symbols(str(tmp_path)) \
        == len(symbols) * len(symbol_data)
    assert all(not path.exists() for path in symbols)
    for name, data in keep.items():
        assert (tmp_path / name).read_bytes() == data


def test_prune_pyside6_runs_the_prereqs_script(clean_app, tmp_path):
    # Without PySide6 the script succeeds without removing anything
    assert clean_app.prune_pyside6(str(tmp_path)) is True


def test_is_fat_macho_of_a_removed_file(clean_app, tmp_path):
    assert clean_app.is_fat_macho(str(tmp_path / "removed")) is False


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
