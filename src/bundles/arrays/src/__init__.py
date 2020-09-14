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

class _ArraysAPI(BundleAPI):
    pass

bundle_api = _ArraysAPI()

_libarrays = None
def load_libarrays():
    '''
    Load the libarrays C++ dynamic library.
    This allows other C modules that link against libarrays to find it
    without setting search paths like RPATH since it will be part of
    the process once loaded and the runtime linker will find it.
    Matching libraries that are part of the process are used on
    macOS, Windows, Linux.
    '''
    global _libarrays
    if _libarrays is None:
        add_library_search_path()
        from . import _arrays
        _libarrays = True

def add_library_search_path():
    import sys
    if sys.platform.startswith('win'):
        import os
        try:
            paths = os.environ['PATH'].split(';')
        except KeyError:
            paths = []
        from os.path import join, dirname
        libdir = join(dirname(__file__), 'lib')
        if libdir in paths:
            return
        paths.append(libdir)
        os.environ['PATH'] = ';'.join(paths)
