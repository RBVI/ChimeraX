#!/usr/bin/env python3
from __future__ import print_function


def make_dependencies(dir_path, output_name):
    import os
    from xml.dom import minidom

    if not dir_path:
        dir_path = '.'

    # Construct mappings from directory name to bundle dependencies
    # and bundle name to directory name
    dependencies = {}
    bundle2dirname = {'ChimeraX-Core': 'core'}
    for dir_name in os.listdir(dir_path):
        p = os.path.join(dir_path, dir_name, "bundle_info.xml")
        if not os.path.exists(p):
            p = os.path.join(dir_path, dir_name, "bundle_info.xml.in")
            if not os.path.exists(p):
                continue
        doc = minidom.parse(p)
        bundle_tags = doc.getElementsByTagName("BundleInfo")
        if len(bundle_tags) != 1:
            print("%s: found %d BundleInfo tags instead of 1" %
                  (dir_name, len(bundle_tags)))
            continue
        bundle_name = bundle_tags[0].getAttribute("name")
        bundle2dirname[bundle_name] = dir_name
        dependencies[dir_name] = deps = ["ChimeraX-Core"]
        for e in doc.getElementsByTagName("Dependency"):
            build_dep = e.getAttribute("build")
            if not build_dep or build_dep.lower() == "false":
                continue
            dep_name = e.getAttribute("name")
            deps.append(dep_name)

    # Loop over all directories and emit one dependency line each
    missing = set()
    clean = {}
    with open(os.path.join(dir_path, output_name), "w") as f:
        for dir_name in sorted(dependencies.keys()):
            dep_dirs = []
            for dep in dependencies[dir_name]:
                try:
                    dep_dir = bundle2dirname[dep]
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
    import os.path
    make_dependencies(os.path.dirname(__file__), "Makefile.dependencies")
