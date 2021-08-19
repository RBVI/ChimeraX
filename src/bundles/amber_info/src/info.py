# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

import os, chimerax
for dir_entry in os.listdir(chimerax.app_bin_dir):
    if not dir_entry.startswith("amber"):
        continue
    try:
        amber_version = int(dir_entry[5:])
    except ValueError:
        continue
    break
else:
    raise AssertionError("No amberXX subdirectory for %s" % chimerax.app_bin_dir)

# antechamber uses system() a lot, and cygwin's implmentation doesn't
# cotton to backslashes as path separators
amber_home = os.path.join(chimerax.app_bin_dir, dir_entry).replace('\\', '/')
amber_bin = amber_home + "/bin"

import sys
if sys.platform == "win32":
    # shut up Cygwin DOS-path complaints
    # and prevent "qm_theory='AM1,", from being changed to "qm_theory=AM1,"
    os.environ['CYGWIN'] = "nodosfilewarning noglob"
    # if user installed Cygwin, use their libs to avoid conflict
    try:
        import winreg
        with winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Cygwin\\setup") as h:
            os.environ['PATH'] = os.environ['PATH'] + ';' + winreg.QueryValueEx(h, "rootdir")[0] + '\\bin\\'
    except WindowsError:
        # Cygwin not installed
        pass
