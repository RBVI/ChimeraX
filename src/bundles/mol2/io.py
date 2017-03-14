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

from chimerax.core.atomic import Atom, Atoms
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
def write_mol2_sort(a1, a2, res_indices=None):
    try:
        ri1 = res_indices[a1.residue]
    except KeyError:
        ri1 = res_indices[a1.residue] = a1.structure.residues.index(a1.residue)
    try:
        ri2 = res_indices[a2.residue]
    except KeyError:
        ri2 = res_indices[a2.residue] = a2.structure.residues.index(a2.residue)
    return cmp(ri1, ri2) or cmp(a1.coord_index, a2.coord_index)

def write_mol2(structures, file_name, status=None, anchor=None, rel_model=None,
        hyd_naming_style="sybyl", multimodel_handling="individual",
        skip=None, res_num=True, gaff_type=False, gaff_fail_error=None,
        temporary=False):
    """Write a Mol2 file.

    Parameters
    ----------

    structures : a list/tuple/set of :py:class:`~chimerax.core.atomic.Structure`s or a single :py:class:`~chimerax.core.atomic.Structure`
        The structure(s)s to write out.

    file_name : str or file object open for writing
        Output file.

    status : function or None
        If not None, a function that takes a string -- used to report the progress of the write.

    anchor : :py:class:`~chimerax.core.atomic.Atoms` collection
        Atoms (and their implied internal bonds) that should be written out to the
        @SET section of the file as the rigid framework for flexible ligand docking.

    hyd_naming_style : "sybyl" or "pdb"
        Controls whether hydrogen names should be "Sybyl-like" (value: sybyl) or
        "PDB-like" (value: pdb) -- e.g.  HG21 vs. 1HG2.

    multimodel_handling : "combined" or "individual"
        Controls whether multiple structures will be combined into a single @MOLECULE
        section (value: combined) or each given its own section (value: individual).

    skip : list/set of :py:class:`~chimerax.core.atomic.Atom`s or an :py:class:`~chimerax.core.atomic.Atoms` collection or None
       Atoms to not output

    res_num : bool
        Controls whether residue sequence numbers are included in the substructure name.
        Since Sybyl Mol2 files include them, this defaults to True.

    gaff_type : bool
       If 'gaff_type' is True, outout GAFF atom types instead of Sybyl atom types.
       `gaff_fail_error`, if specified, is the type of error to throw (e.g. UserError)
       if there is no gaff_type attribute for an atom, otherwise throw the standard AttributeError.

    temporary : bool
       If 'temporary' is True, don't enter the file name into the file history.
    """

    from chimerax.core import io
    f = io.open(file_name, "w")

    sort_func = serial_sort = lambda a1, a2, ri={}: write_mol2_sort(a1, a2, res_indices=ri)

    from chimerax.core.atomic import Structure, Structures
    if isinstance(structures, Structure):
        structures = [structures]
    elif isinstance(structures, Structures):
        class Jumbo:
            def __init__(self, structs):
                self.atoms = structs.atoms
                self.residues = structs.residues
                self.bonds = structs.inter_bonds
                self.name = "(multiple structures)"
        structures = [Jumbo(sel)]
        sort_func = lambda a1, a2: cmp(a1.structure.id, a2.structure.id) or serial_sort(a1, a2)
        multimodel_handling = "individual"

    # transform...
    if rel_model is None:
        from chimerax.core.geometry import identity
        xform = identity()
    else:
        xform = rel_model.scene_position.inverse()

    #TODO
    # need to find amide moieties since Sybyl has an explicit amide type
    if status:
        status("Finding amides")
    from ChemGroup import findGroup
    amides = findGroup("amide", structures)
    amideNs = dict.fromkeys([amide[2] for amide in amides])
    amideCNs = dict.fromkeys([amide[0] for amide in amides])
    amideCNs.update(amideNs)
    amideOs = dict.fromkeys([amide[1] for amide in amides])

    substructureNames = None
    if multimodel_handling == "combined":
        # create a fictitious jumbo model
        class Jumbo:
            def __init__(self, structures):
                self.atoms = []
                self.residues = []
                self.bonds = []
                self.name = structures[0].name + " (combined)"
                for m in structures:
                    self.atoms.extend(m.atoms)
                    self.residues.extend(m.residues)
                    self.bonds.extend(m.bonds)
                # if combining single-residue structures,
                # can be more informative to use model name
                # instead of residue type for substructure
                if len(structures) == len(self.residues):
                    rtypes = [r.type for r in self.residues]
                    if len(set(rtypes)) < len(rtypes):
                        mnames = [m.name for m in structures]
                        if len(set(mnames)) == len(mnames):
                            self.substructureNames = dict(
                                zip(self.residues, mnames))
        structures = [Jumbo(structures)]
        if hasattr(structures[-1], 'substructureNames'):
            substructureNames = structures[-1].substructureNames
            delattr(structures[-1], 'substructureNames')
        sort_func = lambda a1, a2: cmp(a1.structure.id, a2.structure.id) \
            or cmp(a1.structure.subid, a2.structure.subid) \
            or serial_sort(a1, a2)

    # write out structures
    for struct in structures:
        if hasattr(struct, 'mol2comments'):
            for m2c in struct.mol2comments:
                print>>f, m2c
        if hasattr(struct, 'solventInfo' ):
            print>>f, struct.solventInfo

        # molecule section header
        print>>f, "%s" % MOLECULE_HEADER

        # molecule name
        print>>f, "%s" % struct.name

        ATOM_LIST = struct.atoms
        BOND_LIST = struct.bonds
        #TODO: add metal-coordination bonds
        if skip:
            skip = set(skip)
            ATOM_LIST = [a for a in ATOM_LIST if a not in skip]
            BOND_LIST = [b for b in BOND_LIST
                    if b.atoms[0] not in skip
                    and b.atoms[1] not in skip]
        RES_LIST  = struct.residues

        # Chimera has an unusual internal order for its atoms, so
        # sort them by input order
        if status:
            status("Putting atoms in input order")
        ATOM_LIST.sort(sort_func)

        # if anchor is not None, then there will be two entries in
        # the @SET section of the file...
        if anchor:
            sets = 2
        else:
            sets = 0
        # number of entries for various sections...
        print>>f, "%d %d %d 0 %d" % (len(ATOM_LIST), len(BOND_LIST),
                            len(RES_LIST), sets)

        # type of molecule
        if hasattr(struct, "mol2type"):
            mtype = struct.mol2type
        else:
            mtype = "SMALL"
            from chimera.resCode import nucleic3to1, protein3to1
            for r in struct.residues:
                if r.type in protein3to1:
                    mtype = "PROTEIN"
                    break
                if r.type in nucleic3to1:
                    mtype = "NUCLEIC_ACID"
                    break
        print>>f, mtype

        # indicate type of charge information
        if hasattr(struct, 'chargeModel'):
            print>>f, struct.chargeModel
        else:
            print>>f, "NO_CHARGES"

        if hasattr(struct, 'mol2comment'):
            print>>f, "\n%s" % struct.mol2comment
        else:
            print>>f, "\n"


        if status:
            status("writing atoms")
        # atom section header
        print>>f, "%s" % ATOM_HEADER

        # make a dictionary of residue indices so that we can do
        # quick look ups
        res_indices = {}
        for i, r in enumerate(RES_LIST):
            res_indices[r] = i+1
        for i, atom in enumerate(ATOM_LIST):
            # atom ID, starting from 1
            print>>f, "%7d" % (i+1),

            # atom name, possibly rearranged if it's a hydrogen
            if hyd_naming_style == "sybyl" \
                        and not atom.name[0].isalpha():
                atomName = atom.name[1:] + atom.name[0]
            else:
                atomName = atom.name
            print>>f, "%-8s" % atomName,

            # untransformed coordinate position
            coord = xform.apply(atom.xformCoord())
            print>>f, "%9.4f %9.4f %9.4f" % (
                        coord.x, coord.y, coord.z),

            # atom type
            if gaff_type:
                try:
                    atomType = atom.gaff_type
                except AttributeError:
                    if not gaff_fail_error:
                        raise
                    raise gaff_fail_error("%s has no Amber/GAFF type assigned.\n"
                        "Use the AddCharge tool to assign Amber/GAFF types."
                        % atom)
            elif hasattr(atom, 'mol2type'):
                atomType = atom.mol2type
            elif atom in amideNs:
                atomType = "N.am"
            elif atom.residue.id.chainId == "water":
                if atom.element.name == "O":
                    atomType = "O.t3p"
                else:
                    atomType = "H.t3p"
            elif atom.element.name == "N" and len(
            [r for r in atom.minimumRings() if r.aromatic()]) > 0:
                atomType = "N.ar"
            elif atom.idatmType == "C2" and len([nb for nb in atom.neighbors
                                            if nb.idatmType == "Ng+"]) > 2:
                atomType = "C.cat"
            elif sulfurOxygen(atom):
                atomType = "O.2"
            else:
                try:
                    atomType = chimera_to_sybyl[atom.idatmType]
                except KeyError:
                    chimera.replyobj.warning("Atom whose"
                        " IDATM type has no equivalent"
                        " Sybyl type: %s (type: %s)\n"
                        % (atom.oslIdent(),
                        atom.idatmType))
                    atomType = str(atom.element)
            print>>f, "%-5s" % atomType,

            # residue-related info
            res = atom.residue

            # residue index
            print>>f, "%5d" % res_indices[res],

            # substructure identifier and charge
            if hasattr(atom, 'charge') and atom.charge is not None:
                charge = atom.charge
            else:
                charge = 0.0
            if substructureNames:
                rname = substructureNames[res]
            elif res_num:
                rname = "%3s%-5d" % (res.type, res.id.position)
            else:
                rname = "%3s" % res.type
            print>>f, "%s %9.4f" % (rname, charge)


        if status:
            status("writing bonds")
        # bond section header
        print>>f, "%s" % BOND_HEADER


        # make an atom-index dictionary to speed lookups
        atomIndices = {}
        for i, a in enumerate(ATOM_LIST):
            atomIndices[a] = i+1
        for i, bond in enumerate(BOND_LIST):
            a1, a2 = bond.atoms

            # ID
            print>>f, "%6d" % (i+1),

            # atom IDs
            print>>f, "%4d %4d" % (
                    atomIndices[a1], atomIndices[a2]),

            # bond order; give it our best shot...
            if hasattr(bond, 'mol2type'):
                print>>f, bond.mol2type
                continue
            amideA1 = a1 in amideCNs
            amideA2 = a2 in amideCNs
            if amideA1 and amideA2:
                print>>f, "am"
                continue
            if amideA1 or amideA2:
                if a1 in amideOs or a2 in amideOs:
                    print>>f, "2"
                else:
                    print>>f, "1"
                continue
                
            aromatic = False
            #TODO: 'bond' might be a metal-coordination bond, so
            # do an if/else to get the rings
            for ring in bond.minimumRings():
                if ring.aromatic():
                    aromatic = True
                    break
            if aromatic:
                print>>f, "ar"
                continue

            try:
                geom1 = idatm_info[a1.idatmType].geometry
            except KeyError:
                print>>f, "1"
                continue
            try:
                geom2 = idatm_info[a2.idatmType].geometry
            except KeyError:
                print>>f, "1"
                continue
            # sulfone/sulfoxide is classically depicted as double-
            # bonded despite the high dipolar character of the
            # bond making it have single-bond character.  For
            # output, use the classical values.
            if sulfurOxygen(a1) or sulfurOxygen(a2):
                print>>f, "2"
                continue
            if geom1 not in [2,3] or geom2 not in [2,3]:
                print>>f, "1"
                continue
            # if either endpoint atom is in an aromatic ring and
            # the bond isn't, it's a single bond...
            for endp in [a1, a2]:
                aromatic = False
                for ring in endp.minimumRings():
                    if ring.aromatic():
                        aromatic = True
                        break
                if aromatic:
                    break
            else:
                # neither endpoint in aromatic ring
                if geom1 == 2 and geom2 == 2:
                    print>>f, "3"
                else:
                    print>>f, "2"
                continue
            print>>f, "1"

        if status:
            status("writing residues")
        # residue section header
        print>>f, "%s" % SUBSTR_HEADER

        for i, res in enumerate(RES_LIST):
            # residue id field
            print>>f, "%6d" % (i+1),

            # residue name field
            if substructureNames:
                rname = substructureNames[res]
            elif res_num:
                rname = "%3s%-4d" % (res.type, res.id.position)
            else:
                rname = "%3s" % res.type
            print>>f, rname,

            # ID of the root atom of the residue
            from chimera.misc import principalAtom
            chainAtom = principalAtom(res)
            if chainAtom is None:
                if hasattr(res, 'atomsMap'):
                    chainAtom = res.atoms[0]
                else:
                    chainAtom = res.atoms.values()[0][0]
            print>>f, "%5d" % atomIndices[chainAtom],


            print>>f, "RESIDUE           4",

            # Sybyl seems to use chain 'A' when chain ID is blank,
            # so run with that
            chainID = res.id.chainId
            if len(chainID.strip()) != 1:
                chainID = 'A'
            print>>f, "%s     %3s" % (chainID, res.type),

            # number of out-of-substructure bonds
            crossResBonds = 0
            if hasattr(res, "atomsMap"):
                atoms = res.atoms
                for a in atoms:
                    for oa in a.bondsMap.keys():
                        if oa.residue != res:
                            crossResBonds += 1
            else:
                atoms = [a for aList in res.atoms.values()
                            for a in aList]
                for a in atoms:
                    for oa in a.bonds.keys():
                        if oa.residue != res:
                            crossResBonds += 1
            print>>f, "%5d" % crossResBonds,
            # print "ROOT" if first or only residue of a chain
            if a.structure.rootForAtom(a, True).atom.residue == res:
                print>>f, "ROOT"
            else:
                print>>f

        # write flexible ligand docking info
        if anchor:
            if status:
                status("writing anchor info")
            print>>f, "%s" % SET_HEADER
            atomIndices = {}
            for i, a in enumerate(ATOM_LIST):
                atomIndices[a] = i+1
            bondIndices = {}
            for i, b in enumerate(BOND_LIST):
                bondIndices[b] = i+1
            print>>f, "ANCHOR          STATIC     ATOMS    <user>   **** Anchor Atom Set"
            atoms = anchor.atoms()
            print>>f, len(atoms),
            for a in atoms:
                if a in atomIndices:
                    print>>f, atomIndices[a],
            print>>f

            print>>f, "RIGID           STATIC     BONDS    <user>   **** Rigid Bond Set"
            bonds = anchor.bonds()
            print>>f, len(bonds),
            for b in bonds:
                if b in bondIndices:
                    print>>f, bondIndices[b],
            print>>f

    if needClose:
        f.close()

    if not temporary:
        from chimera import triggers
        triggers.activateTrigger('file save', (file_name, 'Mol2'))

def sulfurOxygen(atom):
    if atom.idatmType != "O3-":
        return False
    try:
        s = atom.bondsMap.keys()[0]
    except IndexError:
        return False
    if s.idatmType in ['Son', 'Sxd']:
        return True
    if s.idatmType == 'Sac':
        o3s = [a for a in s.neighbors if a.idatmType == 'O3-']
        o3s.sort()
        return o3s.index(atom) > 1
    return False
