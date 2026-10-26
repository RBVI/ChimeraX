"""Check early bundle_builder installation without deferred tree cleaning."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("tree_clean", [False, True])
@pytest.mark.parametrize("no_prebuilt", [False, True])
def test_prereqs_install_without_tree_clean(tmp_path, tree_clean, no_prebuilt):
    make = shutil.which("gmake") or shutil.which("make")
    if make is None:
        pytest.skip("GNU make is not installed")

    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "Makefile", tmp_path / "Makefile")
    # Keep the real target graph, replacing platform configuration and all
    # recursive makes so the test cannot build or clean the application.
    (tmp_path / "mk").mkdir()
    for name in ("config.make", "subdir.make", "detectOS.make"):
        (tmp_path / "mk" / name).touch()
    recorder = tmp_path / "record.py"
    recorder.write_text('''\
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
args = sys.argv[1:]
if args == ["-C", "prereqs", "install"] or args == ["-C", "prereqs", "install-prebuilt"]:
    (root / "prereqs.done").write_text(args[-1])
elif args == ["-C", "prereqs", "Python.app-install"]:
    assert (root / "prereqs.done").exists()
    (root / "python.done").touch()
else:
    assert (root / "python.done").exists()
    if args == ["-C", "prereqs", "app-install", "SKIP_PYTHON_APP_INSTALL=1"]:
        (root / "app.done").touch()
    else:
        assert args == ["-C", "src/bundles/bundle_builder", "build"]
        (root / "builder.done").touch()
''')
    record = f"{shlex.quote(sys.executable)} {shlex.quote(str(recorder))}"
    env = dict(os.environ)
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES",
                 "TREE_CLEAN", "NO_PREBUILT"):
        env.pop(name, None)
    subprocess.run(
        [make, "--no-print-directory", "-j2", f"MAKE={record}",
         "prereqs-install", *(["TREE_CLEAN=1"] if tree_clean else []),
         *(["NO_PREBUILT=1"] if no_prebuilt else [])],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
        timeout=20,
    )
    assert (tmp_path / "prereqs.done").read_text() == (
        "install" if no_prebuilt else "install-prebuilt"
    )
    assert (tmp_path / "app.done").exists()
    assert (tmp_path / "builder.done").exists()
