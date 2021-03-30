# vim: set expandtab ts=4 sw=4:

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

from chimerax.core.toolshed import BundleAPI

class _AtomicLibAPI(BundleAPI):
    pass

bundle_api = _AtomicLibAPI()

# Load libarrays since atomic_libs C++ shared libraries use it.
from chimerax import arrays
arrays.load_libarrays()

# Include atomic_libs/lib in runtime library search path.
import sys
if sys.platform.startswith('win'):
    from os import path, add_dll_directory
    libdir = path.join(path.dirname(__file__), 'lib')
    add_dll_directory(libdir)

# Load atomic_libs libraries so they are found by other C++ modules that link to them.
from . import _load_libs
