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

from chimerax.core.errors import UserError

def write_dms(session, file_name, *, surface=None, displayed_only=True, save_normals=True, status=None):
    """Write a DMS file.

    Parameters
    ----------

    file_name : str, or file object open for writing
        Output file.

    surface : a MolecularSurface model to write out.  If None, then look for open MolecularSurfaces.
        If there is exactly one, write it out -- otherwise raise an error.

    status : function or None
        If not None, a function that takes a string -- used to report the progress of the write.

    displayed_only : bool
        Whether to limit the output to just the displayed part of the surface.

    save_normals : bool
        Whether to also save vector normals in the file.
    """

    if surface is None:
        from chimerax.atomic import MolecularSurface
        surfs = [m for m in session.models is isinstance(m, MolecularSurface)]
        if len(surfs) != 1:
            raise UserError("If molecular surface not specified, there must be exactly one open;"
                " there are %d" % len(surfs))
        surface = surfs[0]

    if status:
        status("Writing DMS file %s; assigning vertex types" % file_name)

    assign_vertex_types(session, surface, status)

    from chimerax import io
    f = io.open_output(file_name, "utf-8")

    # write out structures
    for struct in structures:
        if hasattr(struct, 'mol2_comments'):
            for m2c in struct.mol2_comments:
                print(m2c, file=f)
        if hasattr(struct, 'solvent_info' ):
            print(struct.solvent_info, file=f)

        # molecule section header
        print("%s" % MOLECULE_HEADER, file=f)

        # molecule name
        print("%s" % struct.name, file=f)

        atoms = list(struct.atoms)
        bonds = list(struct.bonds)
        # add metal-coordination bonds
        coord_grp = struct.pbg_map.get(Structure.PBG_METAL_COORDINATION, None)
        if coord_grp:
            bonds.extend(list(coord_grp.pseudobonds))
        if skip_atoms:
            skip_atoms = set(skip_atoms)
            atoms = [a for a in atoms if a not in skip_atoms]
            bonds = [b for b in bonds
                if b.atoms[0] not in skip_atoms and b.atoms[1] not in skip_atoms]
        residues  = struct.residues

        # Put the atoms in the order we want for output
        if status:
            status("Putting atoms in input order")
        atoms.sort(key=sort_key_func)

        # if anchor is not None, then there will be two entries in
        # the @SET section of the file...
        if anchor:
            sets = 2
        else:
            sets = 0
        # number of entries for various sections...
        print("%d %d %d 0 %d" % (len(atoms), len(bonds), len(residues), sets), file=f)

        # type of molecule
        if hasattr(struct, "mol2_type"):
            mtype = struct.mol2_type
        else:
            mtype = "SMALL"
            from chimerax.atomic import Sequence
            for r in struct.residues:
                if Sequence.protein3to1(r.name) != 'X':
                    mtype = "PROTEIN"
                    break
                if Sequence.nucleic3to1(r.name) != 'X':
                    mtype = "NUCLEIC_ACID"
                    break
        print(mtype, file=f)

        # indicate type of charge information
        if hasattr(struct, 'charge_model'):
            print(struct.charge_model, file=f)
        else:
            print("NO_CHARGES", file=f)

        if hasattr(struct, 'mol2_comment'):
            print("\n%s" % struct.mol2_comment, file=f)
        else:
            print("\n", file=f)


        if status:
            status("writing atoms")
        # atom section header
        print("%s" % ATOM_HEADER, file=f)

        # make a dictionary of residue indices so that we can do quick look ups
        res_indices = {}
        for i, r in enumerate(residues):
            res_indices[r] = i+1
        # only output Mol2 type based on mol2_type attribute if _all_ the heavy atoms
        # have that attribute, and the type corresponds to the current element [#9604]
        for a in atoms:
            if hasattr(a, 'mol2_type'):
                if not a.mol2_type.startswith(a.element.name):
                    types_valid = False
                    break
            elif a.element.number > 1:
                types_valid = False
                break
        else:
            types_valid = True
        for i, atom in enumerate(atoms):
            # atom ID, starting from 1
            print("%7d" % (i+1), end=" ", file=f)

            # atom name, possibly rearranged if it's a hydrogen
            if sybyl_hyd_naming and not atom.name[0].isalpha():
                atom_name = atom.name[1:] + atom.name[0]
            else:
                atom_name = atom.name
            print("%-8s" % atom_name, end=" ", file=f)

            # use correct relative coordinate position
            coord = xform * atom.scene_coord
            print("%9.4f %9.4f %9.4f" % tuple(coord), end=" ", file=f)

            # atom type
            if gaff_type:
                try:
                    atom_type = atom.gaff_type
                except AttributeError:
                    if not gaff_fail_error:
                        raise
                    raise gaff_fail_error("%s has no Amber/GAFF type assigned.\n"
                        "Use the Add Charges tool or 'addcharge' command to assign Amber/GAFF types." % atom)
            elif hasattr(atom, 'mol2_type') and types_valid:
                atom_type = atom.mol2_type
            elif atom in amide_Ns:
                atom_type = "N.am"
            elif atom.structure_category == "solvent" \
            and atom.residue.name in Residue.water_res_names:
                if atom.element.name == "O":
                    atom_type = "O.t3p"
                else:
                    atom_type = "H.t3p"
            elif atom.element.name == "N" and len([r for r in atom.rings() if r.aromatic]) > 0:
                atom_type = "N.ar"
            elif atom.idatm_type == "C2" and len([nb for nb in atom.neighbors
                                            if nb.idatm_type == "Ng+"]) > 2:
                atom_type = "C.cat"
            elif sulfur_oxygen(atom):
                atom_type = "O.2"
            else:
                try:
                    atom_type = chimera_to_sybyl[atom.idatm_type]
                except KeyError:
                    session.logger.warning("Atom whose IDATM type has no equivalent"
                        " Sybyl type: %s (type: %s)" % (atom, atom.idatm_type))
                    atom_type = str(atom.element)
            print("%-5s" % atom_type, end=" ", file=f)

            # residue-related info
            res = atom.residue

            # residue index
            print("%5d" % res_indices[res], end=" ", file=f)

            # substructure identifier and charge
            if hasattr(atom, 'charge') and atom.charge is not None:
                charge = atom.charge
            else:
                charge = 0.0
            if substructure_names:
                rname = substructure_names[res]
            elif res_num:
                rname = "%3s%-5d" % (res.name, res.number)
            else:
                rname = "%3s" % res.name
            print("%s %9.4f" % (rname, charge), file=f)


        if status:
            status("writing bonds")
        # bond section header
        print("%s" % BOND_HEADER, file=f)


        # make an atom-index dictionary to speed lookups
        atom_indices = {}
        for i, a in enumerate(atoms):
            atom_indices[a] = i+1
        for i, bond in enumerate(bonds):
            a1, a2 = bond.atoms

            # ID
            print("%6d" % (i+1), end=" ", file=f)

            # atom IDs
            print("%4d %4d" % (atom_indices[a1], atom_indices[a2]), end=" ", file=f)

            # bond order; give it our best shot...
            if hasattr(bond, 'mol2_type'):
                print(bond.mol2_type, file=f)
                continue
            amide_A1 = a1 in amide_CNs
            amide_A2 = a2 in amide_CNs
            if amide_A1 and amide_A2:
                print("am", file=f)
                continue
            if amide_A1 or amide_A2:
                if a1 in amide_Os or a2 in amide_Os:
                    print("2", file=f)
                else:
                    print("1", file=f)
                continue

            aromatic = False
            # 'bond' might be a metal-coordination bond so do a test for rings
            if hasattr(bond, 'rings'):
                for ring in bond.rings():
                    if ring.aromatic:
                        aromatic = True
                        break
            if aromatic:
                print("ar", file=f)
                continue

            try:
                geom1 = idatm_info[a1.idatm_type].geometry
            except KeyError:
                print("1", file=f)
                continue
            try:
                geom2 = idatm_info[a2.idatm_type].geometry
            except KeyError:
                print("1", file=f)
                continue
            # sulfone/sulfoxide is classically depicted as double-
            # bonded despite the high dipolar character of the
            # bond making it have single-bond character.  For
            # output, use the classical values.
            if sulfur_oxygen(a1) or sulfur_oxygen(a2):
                print("2", file=f)
                continue
            if geom1 not in [2,3] or geom2 not in [2,3]:
                print("1", file=f)
                continue
            # if either endpoint atom is in an aromatic ring and
            # the bond isn't, it's a single bond...
            for endp in [a1, a2]:
                aromatic = False
                for ring in endp.rings():
                    if ring.aromatic:
                        aromatic = True
                        break
                if aromatic:
                    break
            else:
                # neither endpoint in aromatic ring
                if geom1 == 2 and geom2 == 2:
                    print("3", file=f)
                else:
                    print("2", file=f)
                continue
            print("1", file=f)

        if status:
            status("writing residues")
        # residue section header
        print("%s" % SUBSTR_HEADER, file=f)

        for i, res in enumerate(residues):
            # ID of the root atom of the residue
            chain_atom = res.principal_atom
            if chain_atom is None or chain_atom not in atom_indices:
                # if writing out a selection, not all residue atoms
                # might be in atom_indices...
                for chain_atom in res.atoms:
                    if chain_atom in atom_indices:
                        break
                else:
                    # no atoms of this residue being written to file
                    continue

            # residue id field
            print("%6d" % (i+1), end=" ", file=f)

            # residue name field
            if substructure_names:
                rname = substructure_names[res]
            elif res_num:
                rname = "%3s%-4d" % (res.name, res.number)
            else:
                rname = "%3s" % res.name
            print(rname, end=" ", file=f)

            print("%5d" % atom_indices[chain_atom], end=" ", file=f)


            print("RESIDUE           4", end=" ", file=f)

            # Sybyl seems to use chain 'A' when chain ID is blank,
            # so run with that
            chain_id = res.chain_id
            if not chain_id.strip():
                chain_id = 'A'
            print("%-4s  %3s" % (chain_id, res.name), end=" ", file=f)

            # number of out-of-substructure bonds
            cross_res_bonds = 0
            for a in res.atoms:
                for nb in a.neighbors:
                    if nb.residue != res:
                        cross_res_bonds += 1
            print("%5d" % cross_res_bonds, end="", file=f)
            # print "ROOT" if first or only residue of a chain
            if not res.chain or res.chain.existing_residues[0] == res:
                print(" ROOT", file=f)
            else:
                print(file=f)

        # write flexible ligand docking info
        if anchor:
            if status:
                status("writing anchor info")
            print("%s" % SET_HEADER, file=f)
            atom_indices = {}
            for i, a in enumerate(atoms):
                atom_indices[a] = i+1
            bond_indices = {}
            for i, b in enumerate(bonds):
                bond_indices[b] = i+1
            print("ANCHOR          STATIC     ATOMS    <user>   **** Anchor Atom Set", file=f)
            print(len(anchor), end=" ", file=f)
            for a in anchor:
                if a in atom_indices:
                    print(atom_indices[a], end=" ", file=f)
            print(file=f)

            print("RIGID           STATIC     BONDS    <user>   **** Rigid Bond Set", file=f)
            bonds = anchor.intra_bonds
            print(len(bonds), end=" ", file=f)
            for b in bonds:
                if b in bond_indices:
                    print(bond_indices[b], end=" ", file=f)
            print(file=f)

    if file_name != f:
        f.close()

    if status:
        status("Wrote DMS file %s" % file_name)

def sulfur_oxygen(atom):
    if atom.idatm_type != "O3-":
        return False
    try:
        s = atom.neighbors[0]
    except IndexError:
        return False
    if s.idatm_type in ['Son', 'Sxd']:
        return True
    if s.idatm_type == 'Sac':
        o3s = [a for a in s.neighbors if a.idatm_type == 'O3-']
        o3s.sort()
        return o3s.index(atom) > 1
    return False
