#!/usr/bin/env python3

import os
from packaging.requirements import Requirement
from collections import defaultdict
from xml.dom import minidom
from typing import Optional
import tomllib
import argparse


def make_dependencies(dir_path):
    if not dir_path:
        dir_path = "."

    # Construct mappings from directory name to bundle dependencies
    # and bundle name to directory name
    dependencies = defaultdict(list)
    reverse_dependencies = defaultdict(list)
    bundle2dirname = {}
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
            for e in doc.getElementsByTagName("Dependency"):
                build_dep = e.getAttribute("build")
                if not build_dep or build_dep.lower() == "false":
                    continue
                dep_name = e.getAttribute("name")
                dependencies[dir_name].append(dep_name)
                reverse_dependencies[dep_name].append(dir_name)
        elif pyproject:
            bundle_toml: Optional[dict] = None
            try:
                with open(p) as f:
                    bundle_toml = tomllib.loads(f.read())
            except Exception as e:
                print(str(e))
            if bundle_toml:
                build_dependencies = bundle_toml["build-system"]["requires"]
                bundle_name = bundle_toml["project"]["name"]
                bundle2dirname[bundle_name] = dir_name
                for dep in build_dependencies:
                    if dep.startswith("ChimeraX") and dep != "ChimeraX-BundleBuilder":
                        # Strip the version because we don't need it for our internal system
                        dep_as_req = Requirement(dep)
                        dependencies[dir_name].append(dep_as_req.name)
                        reverse_dependencies[dep_as_req.name].append(dir_name)
    return bundle2dirname, dependencies, reverse_dependencies


def write_makefile_dependencies(folder_map, dependencies, dir_path, output_name):
    # Loop over all directories and emit one dependency line each
    missing = set()
    clean = {}
    with open(os.path.join(dir_path, output_name), "w") as f:
        for dir_name in sorted(dependencies.keys()):
            dep_dirs = []
            for dep in dependencies[dir_name]:
                try:
                    dep_dir = folder_map[dep]
                except KeyError:
                    missing.add(dep)
                    continue
                dep_dirs.append(f"{dep_dir}.build")
                clean_dirs = clean.setdefault(dep_dir, [])
                clean_dirs.append(f"{dir_name}.clean")
            if dep_dirs:
                print(f"{dir_name}.build: {' '.join(dep_dirs)}", file=f)
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
    bundle_dirs, dependencies, reverse_dependencies = make_dependencies(
        os.path.dirname(__file__)
    )
    write_makefile_dependencies(
        bundle_dirs, dependencies, os.path.dirname(__file__), "Makefile.dependencies"
    )
