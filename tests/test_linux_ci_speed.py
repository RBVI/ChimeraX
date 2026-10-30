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


@pytest.mark.parametrize("name", ["nightly.yml", "release.yml"])
def test_linux_consumers_wait_only_for_shared_build(name):
    jobs = dict(re.findall(r"^  ([\w-]+):\n(.*?)(?=^  [\w-]+:|\Z)",
                           workflow(name), re.MULTILINE | re.DOTALL))
    producer = jobs["build-linux-bundles"]
    assert "uses: ./.github/workflows/linux_bundles_build.yml" in producer
    assert "needs:" not in producer
    for job in ("build-rocky9-docker", "build-rocky10-docker", "build-ubuntu",
                "build-docker", "build-flatpak"):
        assert "needs: [build-linux-bundles]" in jobs[job]
        assert "needs: [build-rocky9-docker]" not in jobs[job]


def test_shared_build_handoff_is_run_scoped_and_retry_stable():
    names = ("linux_bundles_build.yml", "rocky9_build.yml", "rocky10_build.yml",
             "ubuntu_build.yml", "docker_build.yml", "flatpak_build.yml")
    keys = []
    for name in names:
        source = workflow(name)
        keys.append(re.search(r"key: (linux-bundles-x64-.*)", source).group(1))
        if name != names[0]:
            assert "fail-on-cache-miss: true" in source
    assert len(set(keys)) == 1
    for field in ("github.run_id", "inputs.branch", "inputs.release_type"):
        assert field in keys[0]
    assert "github.run_attempt" not in keys[0]
    assert "github.sha" not in keys[0]


def test_packaging_and_docs_are_not_on_shared_build_path():
    source = workflow("linux_bundles_build.yml")
    assert "make -j$(nproc) src.install" in source
    assert "post-src-install" not in source
    assert "pytest" not in source
    assert "-package" not in source
    rocky = workflow("rocky9_build.yml")
    for required in ("post-src-install", "make pytest-app", "daily", "-package",
                     "upload_docs.py", "upload_build.py"):
        assert required in rocky


def test_packaging_retains_high_compression():
    assert "'-Zxz', '-z9'" in (ROOT / "mkubuntu.py").read_text()
    assert "%define _binary_payload w9T.xzdio" in (ROOT / "mkcentos.py").read_text()
