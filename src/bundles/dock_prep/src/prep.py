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

def prep(session, state, callback, memorization, memorize_name, structures, keywords):
    #TODO memorization
    if keywords.get('del_solvent', False):
        session.logger.info("Deleting solvent")
        for s in structures:
            atoms = s.atoms
            atoms.filter(atoms.structure_categories == "solvent").delete()

    if keywords.get('del_ions', False):
        session.logger.info("Deleting non-metal-complex ions")
        for s in structures:
            atoms = s.atoms
            ions = atoms.filter(atoms.structure_categories == "ions")
            pbg = s.pbg_map.get(s.PBG_METAL_COORDINATION, None)
            if pbg:
                pb_atoms1, pb_atoms2 = pbg.pseudobonds.atoms
                ions = ions.subtract(pb_atoms1)
                ions = ions.subtract(pb_atoms2)
            ions.delete()

    if keywords.get('del_alt_locs', False):
        session.logger.info("Deleting non-current alt locs")
        for s in structures:
            s.delete_alt_locs()
