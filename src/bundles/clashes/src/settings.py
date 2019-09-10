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

from chimerax.atomic import AtomicStructure
from chimerax.core.colors import BuiltinColors

defaults = {
    "action_attr": False,
    "action_color": False,
    "action_log": False,
    "action_pseudobonds": True,
    "action_select": False,
    "atom_color": (255, 0, 0, 255),
    "attr_name": "overlap",
    "bond_separation": 4,
    "clash_hbond_allowance": 0.4,
    "clash_threshold": 0.6,
    "contact_hbond_allowance": 0.0,
    "contact_threshold": -0.4,
    "intra_mol": True,
    "intra_res": False,
    "other_atom_color": None,
    "clash_pb_color": BuiltinColors["medium orchid"],
    "contact_pb_color": BuiltinColors["forest green"],
    "clash_pb_radius": 2 * AtomicStructure.default_hbond_radius,
    "contact_pb_radius": AtomicStructure.default_hbond_radius,
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _ClashSettings(Settings):
	EXPLICIT_SAVE = deepcopy(defaults)

def init(session):
    # each SV instance has its own settings instance
    return _ClashSettings(session, "Clashes/Contacts")
