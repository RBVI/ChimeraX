# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.atomic import AtomicStructure
from chimerax.core.colors import BuiltinColors

defaults = {
    "action_attr": False,
    "action_log": False,
    "action_pseudobonds": True,
    "action_select": False,
    "attr_name": "overlap",
    "bond_separation": 4,
    "clash_hbond_allowance": 0.4,
    "clash_threshold": 0.6,
    "contact_hbond_allowance": 0.0,
    "contact_threshold": -0.4,
    "ignore_hidden_models": False,
    "intra_mol": True,
    "intra_res": False,
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
