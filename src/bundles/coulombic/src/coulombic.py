# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def assign_charges(session, uncharged_residues, his_scheme, charge_method, *, status=None):
    from chimerax.atomic import Atom
    Atom.register_attr(session, "charge", "coulombic coloring", attr_type=float)
    by_structure = {}
    for r in uncharged_residues:
        if r.name == 'HIS':
            r._coulombic_his_scheme = his_scheme
        by_structure.setdefault(r.structure, []).append(r)

    missing_heavies = []
    extra_atoms = []
    copy_needed = {}
    for struct, residue_list in list(by_structure.items()):
        from chimerax.atomic import Residues
        by_structure[struct] = residues = Residues(residue_list)
        try:
            missing, extra = check_residues(residues)
        except NoHydError:
            copy_needed[struct] = True
            continue
        heavies = [info for info in missing if not info[1].startswith('H')]
        missing_heavies.extend(heavies)
        copy_needed[struct] = len(heavies) < len(missing)
        extra_atoms.extend(extra)

    # add_charge will complain about extra heavies, and it also allows any kind of crazy
    # naming scheme for hydrogens, so don't need to do anything with "extra_atoms" anymore

    if missing_heavies:
        from chimerax.core.commands import commas
        if len(missing_heavies) <= 10:
            msg_text = "heavy (non-hydrogen) atoms are missing"
            missing_text = '<br>'.join([str(r) + ' ' + an for r, an in missing_heavies])
        else:
            msg_text = "residues are missing heavy (non-hydrogen) atoms"
            incomplete_residues = sorted(list(set([r for r, an in missing_heavies])))
            missing_text = '<br>'.join([str(r) for r in incomplete_residues])
        session.logger.warning("The following %s, which may result in inaccurate electrostatics:<br>%s"
            % (msg_text, missing_text), is_html=True)

    from chimerax.atomic.struct_edit import standardize_residues
    from chimerax.add_charge import add_charges, default_standardized
    for struct, struct_residues in by_structure.items():
        # need to standardize residues before any structure copy occurs, so do it ourselves here
        # and prevent add_charge from doing it
        standardize_residues(session, struct_residues, res_types=default_standardized)
        if copy_needed[struct]:
            session.logger.status("Copying %s" % struct, secondary=True)
            charged_struct = struct.copy(name="copy of " + struct.name)
            try:
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
            except BaseException:
                charged_struct.delete()
                session.logger.status("", secondary=True)
                raise
        else:
            charged_struct = struct
            charged_residues = struct_residues

        # assign charges
        try:
            add_charges(session, charged_residues, method=charge_method, status=status,
                standardize_residues=False)
        except BaseException:
            if copy_needed[struct]:
                charged_struct.delete()
            raise

        if copy_needed[struct]:
            try:
                session.logger.status("Copying charges back to %s" % struct, secondary=True)
                need_deletion = []
                for o_r in struct_residues:
                    for o_a in o_r.atoms:
                        c_a = orig_a_to_copy[o_a]
                        if c_a.deleted:
                            # add_charge can delete atoms (e.g. 5' phosphates)
                            need_deletion.append(o_a)
                            continue
                        for nb in c_a.neighbors:
                            if nb.residue == c_a.residue and nb not in copy_a_to_orig:
                                c_a.charge += nb.charge
                        o_a.charge = c_a.charge
                for del_a in need_deletion:
                    struct.delete_atom(del_a)
                session.logger.status("Destroying copy of %s" % struct, secondary=True)
            finally:
                charged_struct.delete()

class NoHydError(ValueError):
    pass

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
            # Check if any hydrogens.  If not, raise error to alert caller to add hydrogens
            for a in r.atoms:
                if a.element.number == 1:
                    break
            else:
                raise NoHydError(r.name)
            continue
        name_to_atom = {}
        for a in r.atoms:
            aname = a.name if r.name != "MSE" or a.name != "SE" else "SD"
            name_to_atom[aname] = a
        existing_names = set(name_to_atom.keys())
        needed_names = set(res_data.keys())
        if "OXT" in needed_names and "OXT" not in existing_names and not chain.from_seqres:
            needed_names.remove("OXT")
        missing_atoms.extend([(r, n) for n in needed_names - existing_names])
        extra_atoms.extend([name_to_atom[n] for n in (existing_names - needed_names)])

    return missing_atoms, extra_atoms

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
