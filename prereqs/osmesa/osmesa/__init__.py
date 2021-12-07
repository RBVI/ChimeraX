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

def osmesa_library_path():
    import sys
    import os
    # would like to use sysconfig.get_config_var("SHLIB_SUFFIX"),
    # but it only gives the right answer on Linux
    if sys.platform.startswith("linux"):
        suffix = ".so"
    elif sys.platform == "darwin":
        suffix = ".dylib"
    elif sys.platform.startswith("win"):
        suffix = ".dll"
    path = os.path.join(__path__[0], f'libOSMesa{suffix}')
    return path
