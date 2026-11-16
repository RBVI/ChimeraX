#!/usr/bin/env python3
"""Fail when a bundle changed between two revisions without a version bump.

The comparison runs from the merge base of BASE and HEAD to HEAD, so a branch
or push needs one bump per changed bundle no matter how many commits it holds.
"""

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import PurePosixPath

BUNDLES_DIR = "src/bundles"
CONFIG_NAMES = ("pyproject.toml", "bundle_info.xml.in", "bundle_info.xml")
# Core's version is derived from the build by mkbuildinfo.
EXEMPT_BUNDLES = {"src/bundles/core"}
# Tests are not part of the built wheel.
UNSHIPPED_DIRS = {"tests"}

XML_VERSION = re.compile(r"<BundleInfo\b[^>]*?\bversion\s*=\s*[\"']([^\"']+)[\"']", re.S)
# Quotes are optional: animations assigns a bare number.
INIT_VERSION = re.compile(r"^__version__\s*=\s*[\"']?([^\"'\s#]+)", re.M)


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def bundle_configs(rev, cwd=None):
    """Map each bundle root at rev to the config file names it tracks."""
    roots = {}
    listing = git("ls-tree", "-r", "--name-only", "-z", rev, "--", BUNDLES_DIR, cwd=cwd)
    for name in listing.split("\0"):
        path = PurePosixPath(name)
        if path.name not in CONFIG_NAMES:
            continue
        inside = path.parent.relative_to(BUNDLES_DIR).parts
        # Bundles under a tests directory are fixtures of the enclosing bundle.
        if not inside or UNSHIPPED_DIRS.intersection(inside):
            continue
        roots.setdefault(str(path.parent), set()).add(path.name)
    return roots


def owning_bundle(path, roots):
    for parent in PurePosixPath(path).parents:
        if str(parent) in roots:
            return str(parent)
    return None


def read_version(rev, root, config_names, cwd=None):
    def show(name):
        return git("show", f"{rev}:{root}/{name}", cwd=cwd)

    if "pyproject.toml" in config_names:
        config = tomllib.loads(show("pyproject.toml"))
        version = config.get("project", {}).get("version")
        if version:
            return version
        attr = (
            config.get("tool", {}).get("setuptools", {}).get("dynamic", {})
            .get("version", {}).get("attr")
        )
        if attr:
            module = attr.rpartition(".")[0].replace(".", "/")
            match = INIT_VERSION.search(show(f"{module}/__init__.py"))
            if match:
                return match.group(1)
    for name in CONFIG_NAMES[1:]:
        if name in config_names:
            match = XML_VERSION.search(show(name))
            if match:
                return match.group(1)
    return None


def unbumped_bundles(base, head="HEAD", cwd=None):
    """Return {bundle root: version} for changed bundles whose version did not."""
    base = git("merge-base", base, head, cwd=cwd).strip()
    old_roots = bundle_configs(base, cwd)
    new_roots = bundle_configs(head, cwd)
    changed = git(
        "diff", "--name-only", "--no-renames", "-z", base, head, "--", BUNDLES_DIR,
        cwd=cwd,
    )
    bundles = set()
    for name in filter(None, changed.split("\0")):
        root = owning_bundle(name, new_roots)
        if root is None or root in EXEMPT_BUNDLES or root not in old_roots:
            continue
        if PurePosixPath(name).relative_to(root).parts[0] in UNSHIPPED_DIRS:
            continue
        bundles.add(root)
    failures = {}
    for root in sorted(bundles):
        old = read_version(base, root, old_roots[root], cwd)
        new = read_version(head, root, new_roots[root], cwd)
        if new is None or new == old:
            failures[root] = new
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/develop",
                        help="revision to compare against (default: %(default)s)")
    parser.add_argument("--head", default="HEAD",
                        help="revision to check (default: %(default)s)")
    args = parser.parse_args(argv)
    failures = unbumped_bundles(args.base, args.head)
    for root, version in failures.items():
        if version is None:
            print(f"{root}: could not determine the bundle version")
        else:
            print(f"{root}: changed since {args.base} but the version is still {version}")
    if failures:
        print("Bump the version of each bundle listed above.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
