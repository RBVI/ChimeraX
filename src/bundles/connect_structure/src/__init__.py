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

# ensure C++ shared libs are linkable by us
import chimerax.atomic_lib
import chimerax.atom_search_lib

from chimerax.core.toolshed import BundleAPI

class _ConnectStructureBundle(BundleAPI):
    pass

bundle_api = _ConnectStructureBundle()
