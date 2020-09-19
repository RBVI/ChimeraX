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

class _AtomSearchAPI(BundleAPI):
    pass

bundle_api = _AtomSearchAPI()

# make our shared libs linkable by other bundles
import sys
if sys.platform.startswith('win'):
    import os
    try:
        paths = os.environ['PATH'].split(';')
    except KeyError:
        paths = []
    from os.path import join, dirname
    libdir = join(dirname(__file__), 'lib')
    if libdir not in paths:
        paths.append(libdir)
        os.environ['PATH'] = ';'.join(paths)
from . import _load_libs
