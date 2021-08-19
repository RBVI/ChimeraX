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

# ensure atomic_libs C++ shared libs are linkable by us
import chimerax.atomic_lib

from ._dssp import compute_ss

from chimerax.core.toolshed import BundleAPI

class _DsspBundle(BundleAPI):
    pass

bundle_api = _DsspBundle()
