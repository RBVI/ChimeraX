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

from .chem_group import find_group

# the below in case you want to add your own custom group to group_info...
from .chem_group import group_info, N, C, O, H, R, X, \
    single_bond, heavy, non_oxygen_single_bond, RingAtom

from chimerax.core.toolshed import BundleAPI

class ChemGroupAPI(BundleAPI):

    @staticmethod
    def register_selector(selector_name):
        # 'register_selector' is lazily called when selector is referenced
        from . import data
        data.register_selectors()

bundle_api = ChemGroupAPI()
