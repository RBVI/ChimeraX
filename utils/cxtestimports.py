#!/bin/python3
# vi:set expandtab shiftwidth=4:
#
# Try to import every ChimeraX package and module
# except for the blacklisted ones.
#
import pkgutil
import importlib
import chimerax

import sys
sys.path.append('.')
from import_excludes import IMPORT_EXCLUDES

blacklist = IMPORT_EXCLUDES

failed = 0

def fail(name):
    global failed
    failed = 1
    print(f"Failed to import {name}")

for info in pkgutil.walk_packages(chimerax.__path__, prefix=chimerax.__name__ + '.', onerror=fail):
    module_finder, name, is_pkg = info
    if name in blacklist:
        continue
    print(f"Importing {name}")
    try:
        m = importlib.import_module(name)
    except Exception as err:
        print(f"-> ERROR: {err}")
        failed = 1

raise SystemExit(failed)
