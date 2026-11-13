#!/usr/bin/env python3
import os
from packaging.requirements import Requirement
from xml.dom import minidom
import tomllib


def _is_plain(makefile):
    try:
        with open(makefile) as f:
            lines = [
                line.strip() for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except OSError:
        return False
    return lines == ["include ../Makefile.bundle"]


def make_dependencies(dir_path, output_name):
    # remove when toml is part of the standard Python library

    if not dir_path:
        dir_path = "."

    # Construct mappings from directory name to bundle dependencies
    # and bundle name to directory name
    dependencies = {}
    bundle2dirname = {}
    # Directories of bundles with compiled code
    c_dirs = set()
    # Directories of bundles that depend on core in any way, since building
    # compiled code looks up every dependency's include and library directories
    core_dep_dirs = set()
    # Directories whose Makefile adds nothing to Makefile.bundle, so building
    # them is just building their wheel
    plain_dirs = set()
    for dir_name in os.listdir(dir_path):
        bundle_info = True
        pyproject = False
        p = os.path.join(dir_path, dir_name, "bundle_info.xml")
        if not os.path.exists(p):
            p = os.path.join(dir_path, dir_name, "bundle_info.xml.in")
            if not os.path.exists(p):
                bundle_info = False
                p = os.path.join(dir_path, dir_name, "pyproject.toml")
                if not os.path.exists(p):
                    p = os.path.join(dir_path, dir_name, "pyproject.toml.in")
                    if not os.path.exists(p):
                        continue
                pyproject = True
        if _is_plain(os.path.join(dir_path, dir_name, "Makefile")):
            plain_dirs.add(dir_name)
        if bundle_info:
            doc = minidom.parse(p)
            bundle_tags = doc.getElementsByTagName("BundleInfo")
            if len(bundle_tags) != 1:
                print(
                    "%s: found %d BundleInfo tags instead of 1"
                    % (dir_name, len(bundle_tags))
                )
                continue
            bundle_name = bundle_tags[0].getAttribute("name")
            bundle2dirname[bundle_name] = dir_name
            if bundle_tags[0].getAttribute("purePython").lower() == "false" or any(
                doc.getElementsByTagName(tag)
                for tag in ("CModule", "CLibrary", "CExecutable")
            ):
                c_dirs.add(dir_name)
            dependencies[dir_name] = deps = []
            for e in doc.getElementsByTagName("Dependency"):
                if e.getAttribute("name") == "ChimeraX-Core":
                    core_dep_dirs.add(dir_name)
                build_dep = e.getAttribute("build")
                if not build_dep or build_dep.lower() == "false":
                    continue
                dep_name = e.getAttribute("name")
                deps.append(dep_name)
        elif pyproject:
            bundle_toml: dict = None
            try:
                with open(p) as f:
                    bundle_toml = tomllib.loads(f.read())
            except Exception as e:
                print(str(e))
            build_dependencies = bundle_toml["build-system"]["requires"]
            all_deps = build_dependencies + bundle_toml["project"].get("dependencies", [])
            if any(Requirement(d).name == "ChimeraX-Core" for d in all_deps):
                core_dep_dirs.add(dir_name)
            bundle_name = bundle_toml["project"]["name"]
            bundle2dirname[bundle_name] = dir_name
            tool = bundle_toml.get("tool", {})
            cx = bundle_toml.get("chimerax", tool.get("chimerax", {}))
            if not cx.get("pure", True) or any(
                cx.get(key) for key in ("extension", "library", "executable")
            ):
                c_dirs.add(dir_name)
            dependencies[dir_name] = deps = []
            for dep in build_dependencies:
                # Toss the dependency on core because we build it first in all cases
                if (
                    dep.startswith("ChimeraX")
                    # TODO: These should not really be necessary; all bundles should be made the same way
                    # but we special case bundle builder and core
                    and not dep.startswith("ChimeraX-BundleBuilder")
                    and not dep.startswith("ChimeraX-Core")
                ):
                    # Strip the version because we don't need it for our internal system
                    dep_as_req = Requirement(dep)
                    deps.append(dep_as_req.name)
    missing = set()
    clean = {}
    # Directories that have, or are, build time dependencies
    dependents = {}
    for dir_name, deps in dependencies.items():
        for dep in deps:
            dep_dir = bundle2dirname.get(dep)
            if dep_dir is not None:
                dependents.setdefault(dep_dir, set()).add(dir_name)
                dependents.setdefault(dir_name, set())
    heights = {}

    def height(dir_name):
        # Length of the longest chain of builds waiting on this directory
        if dir_name not in heights:
            heights[dir_name] = 1 + max(
                (height(d) for d in dependents[dir_name]), default=0
            )
        return heights[dir_name]

    depths = {}

    def depth(dir_name):
        # Length of the longest chain of builds this directory waits on
        if dir_name not in depths:
            dep_dirs = [
                bundle2dirname[d] for d in dependencies.get(dir_name, [])
                if d in bundle2dirname
            ]
            depths[dir_name] = 1 + max((depth(d) for d in dep_dirs), default=0)
        return depths[dir_name]

    # Loop over all directories and emit one dependency line each
    with open(os.path.join(dir_path, output_name), "w") as f:
        # Longest chains first so make starts them as soon as possible
        build_dep_dirs = sorted(dependents, key=lambda d: (-height(d), d))
        print(f"BUILD_DEP_SUBDIRS = {' '.join(build_dep_dirs)}", file=f)
        # Directories on a longest chain, which bounds how fast the build can be
        longest = max((depth(d) + height(d) for d in dependents), default=0)
        critical_dirs = [d for d in build_dep_dirs if depth(d) + height(d) == longest]
        print(f"CRITICAL_SUBDIRS = {' '.join(critical_dirs)}", file=f)
        # The others, by when they can start.  The early ones wait at most for
        # the first step of a longest chain, so they have until the end of the
        # chain to finish.  Each of the late ones starts partway along it, so
        # it waits for the chain's step at its own depth, instead of competing
        # with that for the cores.
        others = [d for d in build_dep_dirs if d not in critical_dirs]
        early_dirs = [d for d in others if depth(d) <= 2]
        print(f"EARLY_SUBDIRS = {' '.join(early_dirs)}", file=f)
        late_dirs = [d for d in others if depth(d) > 2]
        print(f"LATE_SUBDIRS = {' '.join(late_dirs)}", file=f)
        waits = [
            f"{d}.build:{c}.build"
            for d in late_dirs
            for c in critical_dirs
            if depth(c) == depth(d)
        ]
        print(f"LATE_WAITS = {' '.join(waits)}", file=f)
        print(f"C_SUBDIRS = {' '.join(sorted(c_dirs))}", file=f)
        print(f"CORE_DEP_SUBDIRS = {' '.join(sorted(core_dep_dirs))}", file=f)
        print(f"PLAIN_SUBDIRS = {' '.join(sorted(plain_dirs))}", file=f)
        for dir_name in sorted(dependencies.keys()):
            dep_dirs = []
            uv_dep_dirs = []
            for dep in dependencies[dir_name]:
                try:
                    dep_dir = bundle2dirname[dep]
                except KeyError:
                    missing.add(dep)
                    continue
                dep_dirs.append(f"{dep_dir}.build")
                uv_dep_dirs.append(f"{dep_dir}.uv-build")
                clean_dirs = clean.setdefault(dep_dir, [])
                clean_dirs.append(f"{dir_name}.clean")
            if dep_dirs:
                print(f"{dir_name}.build: {' '.join(dep_dirs)}", file=f)
                print(f"{dir_name}.uv-build: {' '.join(uv_dep_dirs)}", file=f)
        for dir_name in sorted(clean.keys()):
            clean_dirs = clean[dir_name]
            print(f"{dir_name}.clean: {' '.join(clean_dirs)}", file=f)

    # Report any bundle dependencies that is not found
    missing.discard("qtconsole")
    missing.discard("PyAudio")
    missing.discard("SpeechRecognition")
    missing.discard("netifaces")
    missing.discard("pyrealsense2")
    missing.discard("sfftk-rw")
    if missing:
        print("Missing bundles:")
        for dep in sorted(missing):
            print(" ", dep)


if __name__ == "__main__":
    make_dependencies(os.path.dirname(__file__), "Makefile.dependencies")
