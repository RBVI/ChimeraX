"""Exercise the no-fork batch dispatcher without compiling any bundles."""

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def batch_dispatcher(reports):
    # Load only the dispatcher: importing the package would require the built
    # ChimeraX environment and Cython, neither of which this unit test needs.
    path = ROOT / "src/bundles/bundle_builder/src/__main__.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_build_many")
    namespace = {"os": SimpleNamespace(), "sys": sys, "_report": reports.append}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_build_many"]


def test_windows_batch_keeps_builds_isolated_and_forwards_options(monkeypatch):
    calls, reports = [], []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=kwargs["cwd"].encode())

    monkeypatch.setattr(subprocess, "run", run)
    result = batch_dispatcher(reports)(["one", "two"], 2, debug=True, release=True)
    assert result == 0
    assert sorted(kw["cwd"] for _, kw in calls) == ["one", "two"]
    for command, kw in calls:
        assert command == [sys.executable, "-I", "-m", "chimerax.bundle_builder",
                           "--build", ".", "--debug", "--release"]
        assert kw["stdout"] == subprocess.PIPE
        assert kw["stderr"] == subprocess.STDOUT
    assert reports == [("one", 0, b"one"), ("two", 0, b"two")]


def test_windows_batch_reports_failure_and_does_not_hide_other_results(monkeypatch, capsys):
    reports = []

    def run(command, **kwargs):
        return SimpleNamespace(returncode=1 if kwargs["cwd"] == "bad" else 0,
                               stdout=kwargs["cwd"].encode())

    monkeypatch.setattr(subprocess, "run", run)
    result = batch_dispatcher(reports)(["bad", "good"], 2)
    assert result == 1
    assert reports == [("bad", 1, b"bad"), ("good", 0, b"good")]
    assert "Building failed in: bad" in capsys.readouterr().err


def test_plain_bundles_are_batched_on_windows_too():
    source = (ROOT / "src/bundles/Makefile").read_text()
    assignment = "PY_BATCH_SUBDIRS = $(filter $(PLAIN_SUBDIRS),$(PY_SUBDIRS))"
    assert assignment in source
    assert "ifndef WIN32\n" + assignment not in source
    assert "$(filter-out $(PY_BATCH_SUBDIRS),$(PY_SUBDIRS))" in source
    assert "cp -p $(PY_BATCH_SUBDIRS:=/dist/*.whl) $(SYNC_DIR)" in source
