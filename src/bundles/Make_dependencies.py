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
    # Loop over all directories and emit one dependency line each
    with open(os.path.join(dir_path, output_name), "w") as f:
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
