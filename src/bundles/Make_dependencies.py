#!/usr/bin/env python2
from __future__ import print_function

def make_dependencies(dir_path, output_name):
    import os, os.path
    from xml.dom import minidom

    if not dir_path:
        dir_path = '.'

    # Construct mappings from directory name to bundle dependencies
    # and bundle name to directory name
    build_dependencies = {}
    install_dependencies = {}
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
        build_dependencies[dir_name] = build_deps = []
        install_dependencies[dir_name] = install_deps = []
        for e in doc.getElementsByTagName("Dependency"):
            dep_name = e.getAttribute("name")
            build_attr = e.getAttribute("build")
            if build_attr:
                if build_attr == "true":
                    build_deps.append(dep_name)
                    install_deps.append(dep_name)
                elif build_attr == "false":
                    # this is how to avoid Makefile circular dependencies
                    pass
                else:
                    print("'build' Dependency attribute is neither 'true' nor 'false'")
            else:
                install_deps.append(dep_name)

    # Loop over all directories and emit one dependency line each
    missing = set()
    with open(os.path.join(dir_path, output_name), "w") as f:
        for dir_name in sorted(install_dependencies.keys()):
            dep_dirs = []
            nodep_dirs = []
            for dep in install_dependencies[dir_name]:
                try:
                    dep_dirs.append(bundle2dirname[dep] + ".dep-install")
                    if dep in build_dependencies[dir_name]:
                        nodep_dirs.append(bundle2dirname[dep] + ".nodep-install")
                except KeyError:
                    missing.add(dep)
            if dep_dirs:
                print("%s.dep-install: %s" % (dir_name, ' '.join(dep_dirs)), file=f)
                if nodep_dirs:
                    print("%s.nodep-install: %s" % (dir_name, ' '.join(nodep_dirs)), file=f)

    # Report any bundle dependencies that is not found
    missing.discard("ChimeraX-Core")
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
