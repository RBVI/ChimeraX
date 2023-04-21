#!/bin/python3
# vi:set expandtab shiftwidth=4:
#
# Try to import every ChimeraX package and module
# except for the blacklisted ones.
#
import pkgutil
import importlib
import chimerax

from .filter_modules import module_blacklist, fine_blacklist

def test_chimerax_top_level_imports():
    for info in pkgutil.iter_modules(chimerax.__path__, prefix=chimerax.__name__ + '.'):
        module_finder, name, is_pkg = info
        if name in module_blacklist:
            continue
        m = importlib.import_module(name)

def test_all_imports():
    def check_if_true_error(pkg):
        if pkg not in fine_blacklist and pkg not in module_blacklist:
            raise
    for info in pkgutil.walk_packages(chimerax.__path__, prefix=chimerax.__name__ + '.', onerror=check_if_true_error):
        module_finder, name, is_pkg = info
        if name.endswith(".tool") or (name in fine_blacklist) or (name in module_blacklist):
            continue
        m = importlib.import_module(name)
