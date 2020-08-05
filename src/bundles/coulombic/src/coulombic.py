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

class ChargeError(ValueError):
    pass

def assign_charges(session, uncharged_residues, his_scheme):
    by_structure = {}
    for r in uncharged_residues:
        #if r.name == 'HIS':
        #    r._coulombic_his_scheme = his_scheme
        by_structure.setdefault(r.structure, []).append(r)

    missing_heavies = []
    extra_atoms = []
    copy_needed = {}
    for struct, residue_list in list(by_structure.items()):
        from chimerax.atomic import Residues
        by_structure[struct] = residues = Residues(residue_list)
        missing, extra = check_residues(residues)
        heavies = [info for info in missing if not info[1].startswith('H')]
        missing_heavies.extend(heavies)
        copy_needed[struct] = len(heavies) < len(missing)
        extra_atoms.extend(extras)

    if extra_atoms:
        from chimerax.core.commands import commas
        if len(extra_atoms) <= 7:
            atoms_text = commas(extra_atoms, conjunction="and")
        else:
            atoms_text = commas(extra_atoms[:5] + ["%d other atoms" % (len(extra_atoms)-5)],
                conjunction="and")
        if len([a for a in extra_atoms if a.element.number == 1]) == len(extra_atoms):
            hint = "  Try deleting all hydrogens first."
        else:
            hist = ""
        raise ChargeError("Atoms with non-standard names found in standard residues: %s.%s"
            % (atoms_text, hint))

    #TODO: warn about missing heavy atoms
