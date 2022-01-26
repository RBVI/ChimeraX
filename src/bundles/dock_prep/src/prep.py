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

def prep(session, state, callback, memorization, memorize_name, structures, *, del_solvent=False):
    #TODO memorization
    if del_solvent:
        session.logger.info("Deleting solvent")
        for s in structures:
            atoms = s.atoms
            atoms.filter(atoms.structure_categories == "solvent").delete()

