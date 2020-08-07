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
        extra_atoms.extend(extra)

    if extra_atoms:
        from chimerax.core.commands import commas
        if len(extra_atoms) <= 7:
            atoms_text = commas([str(a) for a in extra_atoms], conjunction="and")
        else:
            atoms_text = commas([str(a) for a in extra_atoms[:5]]
                + ["%d other atoms" % (len(extra_atoms)-5)], conjunction="and")
        if len([a for a in extra_atoms if a.element.number == 1]) == len(extra_atoms):
            hint = "  Try deleting all hydrogens first."
        else:
            hint = ""
        raise ChargeError("Atoms with non-standard names found in standard residues: %s.%s"
            % (atoms_text, hint))

    if missing_heavies:
        from chimerax.core.commands import commas
        if len(missing_heavies) <= 7:
            atoms_text = commas([str(r) + ' ' + an for r, an in missing_heavies], conjunction="and")
        else:
            atoms_text = commas([str(r) + ' ' + an for r, an in missing_heavies[:5]]
                + ["%d other atoms" % (len(missing_heavies)-5)], conjunction="and")
        session.logger.warning("The following heavy (non-hydrogen) atoms are missing, which may result"
            " in inaccurate electrostatics: %s" % atoms_text)

    #TODO: treat MSE as MET

def check_residues(residues):
    from .data import starting_residues, ending_residues, other_residues
    missing_atoms = []
    extra_atoms = []
    for r in residues:
        chain = r.chain
        if r.name == "HIS":
            proton_names = set(r.atoms.names[r.atoms.elements.numbers == 1])
            if 'HD1' not in proton_names:
                rname = 'HIE'
            elif 'HE2' not in proton_names:
                rname = 'HID'
            else:
                rname = 'HIP'
        else:
            rname = r.name
        if chain is None:
            reference = other_residues
            type_text = "non-polymeric"
        elif r == chain.residues[0]:
            reference = starting_residues
            type_text = "chain-initial"
        elif r == chain.residues[-1]:
            reference = ending_residues
            type_text = "chain-final"
        else:
            reference = other_residues
            type_text = "mid-chain"
        try:
            res_data = reference[rname]
        except KeyError:
            raise ChargeError("Don't know charges for %s residue %s" % (type_text, r.name))
        name_to_atom = {}
        for a in r.atoms:
            name_to_atom[a.name] = a
        existing_names = set(name_to_atom.keys())
        needed_names = set(res_data.keys())
        missing_atoms.extend([(r, n) for n in needed_names - existing_names])
        extra_atoms.extend([name_to_atom[n] for n in (existing_names - needed_names)])

    return missing_atoms, extra_atoms
