# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2023 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os
import pkgutil
import shutil
import sys
sys.path.append('.')
import chimerax

try:
    from import_excludes import WHEEL_MODULE_EXCLUDES as module_blacklist
    from import_excludes import WHEEL_FINE_EXCLUDES as fine_blacklist
except ImportError:
    _three_up = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    # utils/build/wheel/ -> _three_up is utils/
    sys.path.insert(0, _three_up)
    # build/wheel/ (CI) -> _three_up is repo root, need utils/ under it
    sys.path.insert(0, os.path.join(_three_up, "utils"))
    from import_excludes import WHEEL_MODULE_EXCLUDES as module_blacklist
    from import_excludes import WHEEL_FINE_EXCLUDES as fine_blacklist

if __name__ == "__main__":

    def check_if_true_error(pkg):
        if pkg not in fine_blacklist and pkg not in module_blacklist:
            raise

    kept_files = []
    for info in pkgutil.walk_packages(
        chimerax.__path__, prefix=chimerax.__name__ + ".", onerror=check_if_true_error
    ):
        module_finder, name, is_pkg = info
        if (
            any(name.endswith(x) for x in ["tool", "ui", "cgi"])
            or (name in fine_blacklist)
            or (name in module_blacklist)
        ):
            path_to_thing = os.path.sep.join(name.split("."))
            if os.path.isdir(path_to_thing):
                shutil.rmtree(path_to_thing)
            elif os.path.isfile(path_to_thing + ".py"):
                os.remove(path_to_thing + ".py")
        else:
            path_to_thing = os.path.sep.join(name.split("."))
            kept_files.append(path_to_thing)
    print("\n".join(kept_files))
