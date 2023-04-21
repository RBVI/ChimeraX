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
from ._pyarrays import get_lib, get_include, load_libarrays

__all__ = ["get_lib", "get_include", "load_libarrays"]

# try:
#     from chimerax import running_as_application
# except ImportError:
#     running_as_application = False
# if running_as_application:
from ._pyarrays import bundle_api
__all__.append("bundle_api")
