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
    from chimerax.atomic import Atom
    Atom.register_attr(session, "charge", "coulombic coloring", attr_type=float)
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

    for struct, struct_residues in by_structure.items():
        if copy_needed[struct]:
            session.logger.status("Copying %s" % struct, secondary=True)
            charged_struct = struct.copy(name="copy of " + struct.name)
            orig_a_to_copy = {}
            copy_a_to_orig = {}
            for o_a, c_a in zip(struct.atoms, charged_struct.atoms):
                orig_a_to_copy[o_a] = c_a
                copy_a_to_orig[c_a] = o_a
            orig_r_to_copy = {}
            copy_r_to_orig = {}
            for o_r, c_r in zip(struct.residues, charged_struct.residues):
                orig_r_to_copy[o_r] = c_r
                copy_r_to_orig[c_r] = o_r
            from chimerax.addh.cmd import cmd_addh
            hbond = False
            if his_scheme is None:
                if len(struct_residues[struct_residues.names == "HIS"]) > 0:
                    hbond = True
            from chimerax.atomic import AtomicStructures
            addh_structures = AtomicStructures([charged_struct])
            session.logger.status("Adding hydrogens to copy of %s" % struct, secondary=True)
            session.silent = True
            try:
                cmd_addh(session, addh_structures, hbond=hbond)
            finally:
                session.silent = False
            charged_residues = [orig_r_to_copy[r] for r in struct_residues]
            session.logger.status("Assigning charges to copy of %s" % struct, secondary=True)
        else:
            charged_struct = struct
            charged_residues = struct_residues

        # assign charges
        assign_residue_charges(charged_residues, his_scheme)

        if copy_needed[struct]:
            session.logger.status("Copying charges back to %s" % struct, secondary=True)
            for o_r in struct_residues:
                for o_a in o_r.atoms:
                    c_a = orig_a_to_copy[o_a]
                    for nb in c_a.neighbors:
                        if nb.residue == c_a.residue and nb not in copy_a_to_orig:
                            c_a.charge += nb.charge
                    o_a.charge = c_a.charge
            session.logger.status("Destroying copy of %s" % struct, secondary=True)
            charged_struct.delete()

def check_residues(residues):
    from .data import starting_residues, ending_residues, other_residues
    missing_atoms = []
    extra_atoms = []
    for r in residues:
        chain = r.chain
        rname = template_residue_name(r)
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
            aname = a.name if r.name != "MSE" or a.name != "SE" else "SD"
            name_to_atom[aname] = a
        existing_names = set(name_to_atom.keys())
        needed_names = set(res_data.keys())
        missing_atoms.extend([(r, n) for n in needed_names - existing_names])
        extra_atoms.extend([name_to_atom[n] for n in (existing_names - needed_names)])

    return missing_atoms, extra_atoms

def assign_residue_charges(residues, his_scheme):
    from .data import starting_residues, ending_residues, other_residues
    for r in residues:
        chain = r.chain
        rname = template_residue_name(r)
        if chain is None:
            reference = other_residues
        elif r == chain.residues[0]:
            reference = starting_residues
        elif r == chain.residues[-1]:
            reference = ending_residues
        else:
            reference = other_residues
        res_data = reference[rname]
        for a in r.atoms:
            aname = a.name if r.name != "MSE" or a.name != "SE" else "SD"
            try:
                a.charge = res_data[aname]
            except KeyError:
                print(a)
                raise

def template_residue_name(r):
    if r.name == "HIS":
        proton_names = set(r.atoms.names[r.atoms.elements.numbers == 1])
        if 'HD1' not in proton_names:
            return 'HIE'
        if 'HE2' not in proton_names:
            return 'HID'
        return 'HIP'

    if r.name == "MSE":
        return "MET"

    if r.name == "CYS":
        sulfur = r.find_atom("SG")
        if sulfur:
            for nb in sulfur.neighbors:
                if nb.element.name == 'S':
                    return "CYX"
        return "CYS"

    return r.name
