"""Linux cache isolation and build parallelism, without building ChimeraX."""

import importlib.util
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workflow(name):
    return (ROOT / ".github/workflows" / name).read_text()


@pytest.fixture
def python_cache(tmp_path):
    for name in ("mk/config.make", "prereqs/Python/Makefile",
                 "prereqs/Python/Python.patch", "utils/ci/python_cache_key.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    module = load_module("utils/ci/python_cache_key.py")
    arguments = dict(root=tmp_path, packages="openssl=1\nglibc=2\n",
                     os_release="rocky9", machine="x86_64", compiler="gcc 11",
                     environment={"DEBUG": "", "CFLAGS": ""})
    return module.cache_key, arguments


def test_python_cache_independent_of_package_listing_order(python_cache):
    key, args = python_cache
    assert key(**args) == key(**dict(args, packages="glibc=2\nopenssl=1\n"))


@pytest.mark.parametrize("field,value", [
    ("packages", "openssl=2\nglibc=2\n"), ("os_release", "ubuntu24"),
    ("machine", "aarch64"), ("compiler", "gcc 14"),
    ("environment", {"DEBUG": "1", "CFLAGS": ""}),
])
def test_python_cache_invalidated_by_environment(python_cache, field, value):
    key, args = python_cache
    assert key(**args) != key(**dict(args, **{field: value}))


@pytest.mark.parametrize("name", ["mk/config.make", "prereqs/Python/Makefile",
                                  "prereqs/Python/Python.patch"])
def test_python_cache_invalidated_by_build_recipe(python_cache, name):
    key, args = python_cache
    before = key(**args)
    (args["root"] / name).write_text("changed")
    assert key(**args) != before


def test_flatpak_never_restores_host_python():
    for name in ("flatpak_build.yml", "flatpak_arm_build.yml", "check-for-changes.yml"):
        source = workflow(name)
        flatpak = source[source.index("image: fedora:latest"):]
        cache = flatpak.split("uses: ./utils/ci/consolidated_cache", 1)[1].split("      - ", 1)[0]
        assert "cache_native_python: 'false'" in cache
