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

from chimerax.atomic.struct_edit import standardizable_residues as std_res
defaults = {
    "standardize_residues": std_res,
    "del_solvent": True,
	"del_ions": True,
    "del_alt_locs": True,
    "complete_side_chains": True,
    "ah": True,
    "ac": True,
    "write_mol2": True,
}

from copy import deepcopy

def get_settings(session, memorize_requester, main_settings_name, defaults):
    from  chimerax.core.settings import Settings
    class DP_Settings(Settings):
        AUTO_SAVE = deepcopy(defaults)

    return DP_Settings(session, "%s %s" % (memorize_requester, main_settings_name))
