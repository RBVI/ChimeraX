#!/usr/bin/env python2
from __future__ import print_function

def make_dependencies(dir_path, output_name):
    import os, os.path
    from xml.dom import minidom

    if not dir_path:
        dir_path = '.'

    # Construct mappings from directory name to bundle dependencies
    # and bundle name to directory name
    dependencies = {}
    bundle2dirname = {}
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
        dependencies[dir_name] = deps = []
        for e in doc.getElementsByTagName("Dependency"):
            dep_name = e.getAttribute("name")
            deps.append(dep_name)

    # Loop over all directories and emit one dependency line each
    missing = set()
    with open(os.path.join(dir_path, output_name), "w") as f:
        for dir_name in sorted(dependencies.keys()):
            dep_dirs = []
            for dep in dependencies[dir_name]:
                try:
                    dep_dirs.append(bundle2dirname[dep] + ".install")
                except KeyError:
                    missing.add(dep)
            if dep_dirs:
                print("%s.install: %s" % (dir_name, ' '.join(dep_dirs)), file=f)

    # Report any bundle dependencies that is not found
    missing.discard("ChimeraX-Core")
    missing.discard("qtconsole")
    if missing:
        print("Missing bundles:")
        for dep in sorted(missing):
            print(" ", dep)


if __name__ == "__main__":
    import os.path
    make_dependencies(os.path.dirname(__file__), "Makefile.dependencies")
