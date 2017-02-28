#!/usr/bin/python3
# vim: set expandtab shiftwidth=4 softtabstop=4:

def collect(directory, single=False):
    import os, os.path, sys
    collectors = []
    errors = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for skip in ["build", "unused", "wsgi", "llgr", "hydra"]:
            try:
                dirnames.remove(skip)
            except ValueError:
                pass
        report_dir = dirpath
        if single and dirpath.startswith(directory):
            report_dir = dirpath[len(directory):]
            if report_dir.startswith(os.path.sep):
                report_dir = report_dir[1:]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            report_path = os.path.join(report_dir, filename)
            try:
                c = Collector(path, report_path)
            except Exception as e:
                msg = "%s: %s" % (report_path, str(e))
                errors.append(msg)
            else:
                collectors.append(c)
    return collectors, errors

import ast
class Collector(ast.NodeVisitor):
    def __init__(self, filename, report_name):
        self.filename = filename
        self.report_name = report_name
        self.module_names = set()
        with open(filename, encoding="utf-8") as f:
            root = ast.parse(f.read(), filename=filename, mode="exec")
        self.visit(root)
        all_names = set()
        for name in self.module_names:
            parts = name.split('.')
            for i in range(len(parts)):
                mod_name = '.'.join(parts[:i+1])
                all_names.add(mod_name)
        self.module_names = all_names

    def visit_Import(self, node):
        # import sys
        # print(ast.dump(node), file=sys.stderr)
        for mod in node.names:
            self.module_names.add(mod.name)

    def visit_ImportFrom(self, node):
        # import sys
        # print(ast.dump(node), file=sys.stderr)
        if node.module is None:
            module_name = ""
        else:
            self.module_names.add(node.module)
            module_name = node.module
        for mod in node.names:
            self.module_names.add("%s.%s" % (module_name, mod.name))

def filter_collectors(collectors, pkg):
    return [c for c in collectors if pkg in c.module_names]

def report_importers(importers, pkg, comment=None, f=None):
    if f is None:
        import sys
        f = sys.stdout
    if comment:
        print("%s (%s):" % (pkg, comment), file=f)
    else:
        print("%s:" % pkg, file=f)
    if importers:
        for c in sorted(importers, key=lambda c: c.report_name):
            print("\t%s" % c.report_name, file=f)
    else:
        print("\tNone", file=f)

def report_ood(pkgs, f=None):
    if f is None:
        import sys
        f = sys.stdout
    print("Out of date:", file=f)
    labels = ("Package", "Current", "New")
    name_len = max(len(labels[0]),
                   max([len(pkg["name"]) for pkg in pkgs]))
    current_len = max(len(labels[1]),
                      max([len(pkg["version"]) for pkg in pkgs]))
    new_len = max(len(labels[2]),
                  max([len(pkg["latest_version"]) for pkg in pkgs]))
    fmt = "%%-%ds  %%-%ds  %%-%ds" % (name_len, current_len, new_len)
    print(fmt % labels, file=f)
    for p in pkgs:
        print(fmt % (p["name"], p["version"], p["latest_version"]), file=f)
