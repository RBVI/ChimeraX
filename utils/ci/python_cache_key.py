"""Fingerprint the native Linux environment used to build our CPython archive.

Run after installing system dependencies. Deliberately do not use restore-key
prefixes: a Python linked against an older system library is not a cache hit.
Flatpak must use its SDK identity instead of this host fingerprint.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def cache_key(root, packages, os_release, machine, compiler, environment):
    digest = hashlib.sha256()
    identity = {
        "packages": sorted(packages.splitlines()),
        "os_release": os_release,
        "machine": machine,
        "compiler": compiler,
        # sysconfig records build paths; don't reuse it at a different path.
        "root": str(root.resolve()),
        "environment": environment,
    }
    digest.update(json.dumps(identity, sort_keys=True).encode())
    paths = set((root / "mk").glob("*.make"))
    paths.update((root / "prereqs/Python").glob("Makefile*"))
    paths.update((root / "prereqs/Python").glob("*.patch"))
    paths.add(root / "utils/ci/python_cache_key.py")
    paths.update(root / name for name in (".makerc", "build_type")
                 if (root / name).exists())
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return "linux-python-v1-" + digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    if shutil.which("dpkg-query"):
        command = ["dpkg-query", "-W", "-f=${Package}:${Architecture}=${Version}\n"]
    else:
        command = ["rpm", "-qa", "--qf", "%{NAME}.%{ARCH}=%{EPOCHNUM}:%{VERSION}-%{RELEASE}\n"]
    packages = subprocess.check_output(command, text=True)
    compiler = subprocess.check_output(["gcc", "--version"], text=True)
    environment = {name: os.environ.get(name, "") for name in (
        "CC", "CFLAGS", "CPPFLAGS", "LDFLAGS", "DEBUG", "HARDENING",
        "CONFIG_OPTIONS", "TARGET_ARCH", "BUILD_TYPE",
    )}
    print(cache_key(root, packages, Path("/etc/os-release").read_text(),
                    platform.machine(), compiler, environment))


if __name__ == "__main__":
    main()
