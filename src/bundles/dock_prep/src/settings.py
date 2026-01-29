# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

from chimerax.atomic.struct_edit import standardizable_residues as std_res
defaults = {
    "ac": True,
    "ah": True,
    "complete_side_chains": True,
    "del_alt_locs": True,
    "del_missing_backbone": False,
	"del_ions": True,
    "del_solvent": True,
    "standardize_residues": std_res,
    "write_mol2": True,
}

from copy import deepcopy

def get_settings(session, memorize_requester, main_settings_name, defaults):
    from  chimerax.core.settings import Settings
    class DP_Settings(Settings):
        AUTO_SAVE = deepcopy(defaults)

    return DP_Settings(session, "%s %s" % (memorize_requester, main_settings_name))
