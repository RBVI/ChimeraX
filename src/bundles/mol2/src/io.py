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

from chimerax.atomic import Atom, Atoms
idatm_info = Atom.idatm_info_map

MOLECULE_HEADER = "@<TRIPOS>MOLECULE"
ATOM_HEADER     = "@<TRIPOS>ATOM"
BOND_HEADER     = "@<TRIPOS>BOND"
SUBSTR_HEADER   = "@<TRIPOS>SUBSTRUCTURE"
SET_HEADER    = "@<TRIPOS>SET"


# The 'chimera_to_sybyl' dictionary is used to map ChimeraX atom types to Sybyl
# atom types.
chimera_to_sybyl = {
    'C3'  : 'C.3',
    'C2'  : 'C.2',
    'Car' : 'C.ar',
    'Cac' : 'C.2',
    'C1'  : 'C.1',
    'C1-' : 'C.1',
    'N3+' : 'N.4',
    'N3'  : 'N.3',
    'N2+' : 'N.2',
    'N2'  : 'N.2',
    'Npl' : 'N.pl3',
    'Ng+' : 'N.pl3',
    'Ntr' : 'N.2',
    'N1+' : 'N.1',
    'N1'  : 'N.1',
    'O3'  : 'O.3',
    'O2'  : 'O.2',
    'Oar' : 'O.2',
    'Oar+': 'O.2',
    'O3-' : 'O.co2',
    'O2-' : 'O.co2',
    'O1'  : 'O.2', # no sp oxygen in Sybyl
    'O1+' : 'O.2', # no sp oxygen in Sybyl
    'S3+' : 'S.3',
    'S3-' : 'S.3',
    'S3'  : 'S.3',
    'S2'  : 'S.2',
    'Sar' : 'S.2',
    'Sac' : 'S.o2',
    'Son' : 'S.o2',
    'Sxd' : 'S.o',
    'S'   : 'S.3',
    'Pac' : 'P.3',
    'Pox' : 'P.3',
    'P3+' : 'P.3',
    'P'   : 'P.3',
    'HC'  : 'H',
    'H'   : 'H',
    'DC'  : 'H',
    'D'   : 'H',
    'F'   : 'F',
    'Cl'  : 'Cl',
    'Br'  : 'Br',
    'I'   : 'I',
    'Li'  : 'Li',
    'Na'  : 'Na',
    'Mg'  : 'Mg',
    'Al'  : 'Al',
    'Si'  : 'Si',
    'K'   : 'K',
    'Ca'  : 'Ca',
    'Mn'  : 'Mn',
    'Fe'  : 'Fe',
    'Cu'  : 'Cu',
    'Zn'  : 'Zn',
    'Se'  : 'Se',
    'Mo'  : 'Mo',
    'Sn'  : 'Sn'
}

# keep added hydrogens with their residue while keeping residues in sequence order...
def write_mol2_sort_key(a, res_indices=None):
    try:
        ri = res_indices[a.residue]
    except KeyError:
        ri = res_indices[a.residue] = a.structure.residues.index(a.residue)
    return (ri, a.coord_index)

def write_mol2(session, file_name, *, models=None, atoms=None, status=None, anchor=None,
        rel_model=None, sybyl_hyd_naming=True, combine_models=False,
        skip_atoms=None, res_num=False, gaff_type=False, gaff_fail_error=None):
    """Write a Mol2 file.

    Parameters
    ----------

    file_name : str, or file object open for writing
        Output file.

    models : a list/tuple/set of models (:py:class:`~chimerax.atomic.Structure`s)
        or a single :py:class:`~chimerax.atomic.Structure`
        The structure(s) to write out. If None (and 'atoms' is also None) then
        write out all structures.

    atoms : an :py:class:`~chimerax.atomic.Atoms` collection or None.  If not None,
        then 'models' must be None.

    status : function or None
        If not None, a function that takes a string -- used to report the progress of the write.

    anchor : :py:class:`~chimerax.atomic.Atoms` collection
        Atoms (and their implied internal bonds) that should be written out to the
        @SET section of the file as the rigid framework for flexible ligand docking.

    rel_model : Model whose coordinate system the coordinates should be written out reletive to,
        i.e. take the output atoms' coordinates and apply the inverse of the rel_model's transform.

    sybyl_hyd_naming : bool
        Controls whether hydrogen names should be "Sybyl-like" or "PDB-like" -- e.g.  HG21 vs. 1HG2.

    combine_models : bool
        Controls whether multiple structures will be combined into a single @MOLECULE
        section (value: True) or each given its own section (value: False).

    skip_atoms : list/set of :py:class:`~chimerax.atomic.Atom`s or an :py:class:`~chimerax.atomic.Atoms` collection or None
       Atoms to not output

    res_num : bool
        Controls whether residue sequence numbers are included in the substructure name.
        Since Sybyl Mol2 files include them, this defaults to True.

    gaff_type : bool
       If 'gaff_type' is True, outout GAFF atom types instead of Sybyl atom types.
       `gaff_fail_error`, if specified, is the type of error to throw (e.g. UserError)
       if there is no gaff_type attribute for an atom, otherwise throw the standard AttributeError.
    """

    if status:
        status("Writing Mol2 file %s" % file_name)

    from chimerax import io
    f = io.open_output(file_name, "utf-8")

    sort_key_func = serial_sort_key = lambda a, ri={}: write_mol2_sort_key(a, res_indices=ri)

    from chimerax.atomic import Structure, Atoms, Residue
    class JPBGroup:
        def __init__(self, atoms):
            atom_set = set(atoms)
            pbs = []
            for s in atoms.unique_structures:
                pbg = s.pbg_map.get(s.PBG_METAL_COORDINATION, None)
                if not pbg:
                    continue
                for pb in pbg.pseudobonds:
                    if pb.atoms[0] in atom_set and pb.atoms[1] in atom_set:
                        pbs.append(pb)
            self._pbs = pbs

        @property
        def pseudobonds(self):
            return self._pbs
    if models is None:
        if atoms is None:
            structures = session.models.list(type=Structure)
        else:
            structures = atoms
    else:
        if atoms is None:
            if isinstance(models, Structure):
                structures = [models]
            else:
                structures = [m for m in models if isinstance(m, Structure)]
        else:
            raise ValueError("Cannot specify both 'models' and 'atoms' keywords")

    if isinstance(structures, Atoms):
        class Jumbo:
            def __init__(self, atoms):
                self.atoms = atoms
                self.residues = atoms.unique_residues
                self.bonds = atoms.intra_bonds
                self.name = "(selection)"
                self.pbg_map = { Structure.PBG_METAL_COORDINATION: JPBGroup(atoms) }

        structures = [Jumbo(structures)]
        sort_key_func = lambda a: (a.structure.id,) + serial_sort_key(a)
        combine_models = False

    # transform...
    if rel_model is None:
        from chimerax.geometry import identity
        xform = identity()
    else:
        xform = rel_model.scene_position.inverse()

    # need to find amide moieties since Sybyl has an explicit amide type
    if status:
        status("Finding amides")
    from chimerax.chem_group import find_group
    amides = find_group("amide", structures)
    amide_Ns = set([amide[2] for amide in amides])
    amide_CNs = set([amide[0] for amide in amides])
    amide_CNs.update(amide_Ns)
    amide_Os = set([amide[1] for amide in amides])

    substructure_names = None
    if combine_models and len(structures) > 1:
        # create a fictitious jumbo model
        class Jumbo:
            def __init__(self, structures):
                self.name = structures[0].name + " (combined)"
                from chimerax.atomic import concatenate
                self.atoms = concatenate([s.atoms for s in structures])
                self.bonds = concatenate([s.bonds for s in structures])
                self.residues = concatenate([s.residues for s in structures])
                self.pbg_map = { Structure.PBG_METAL_COORDINATION: JPBGroup(self.atoms) }
                # if combining single-residue structures,
                # can be more informative to use model name
                # instead of residue type for substructure
                if len(structures) == len(self.residues):
                    rnames = self.residues.names
                    if len(set(rnames)) < len(rnames):
                        snames = [s.name for s in structures]
                        if len(set(snames)) == len(snames):
                            self.substructure_names = dict(zip(self.residues, snames))
        structures = [Jumbo(structures)]
        if hasattr(structures[-1], 'substructure_names'):
            substructure_names = structures[-1].substructure_names
            delattr(structures[-1], 'substructure_names')
        sort_key_func = lambda a: (a.structure.id,) + serial_sort_key(a)

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
                        "Use the AddCharge tool to assign Amber/GAFF types." % atom)
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
        status("Wrote Mol2 file %s" % file_name)

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
