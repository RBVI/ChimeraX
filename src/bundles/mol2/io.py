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

from chimerax.core.atomic import Atom
idatm_info = Atom.idatm_info_map
#TODO
from chimera.selection import Selection, ItemizedSelection

MOLECULE_HEADER = "@<TRIPOS>MOLECULE"
ATOM_HEADER     = "@<TRIPOS>ATOM"
BOND_HEADER     = "@<TRIPOS>BOND"
SUBSTR_HEADER   = "@<TRIPOS>SUBSTRUCTURE"
SET_HEADER    = "@<TRIPOS>SET"


# The 'chimera2sybyl' dictionary is used to map Chimera atom types to Sybyl
# atom types.
chimera2sybyl = {
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
def writeMol2Sort(a1, a2, resIndices=None):
    try:
        ri1 = resIndices[a1.residue]
    except KeyError:
        ri1 = resIndices[a1.residue] = a1.molecule.residues.index(a1.residue)
    try:
        ri2 = resIndices[a2.residue]
    except KeyError:
        ri2 = resIndices[a2.residue] = a2.molecule.residues.index(a2.residue)
    return cmp(ri1, ri2) or cmp(a1.coordIndex, a2.coordIndex)


def writeMol2(models, fileName, status=None, anchor=None, relModel=None,
        hydNamingStyle="sybyl", multimodelHandling="individual",
        skip=None, resNum=True, gaffType=False, gaffFailError=None,
        temporary=False):
    """Write a Mol2 file.

       'models' are the models to write out into a file named 'fileName'.

       'status', if not None, is a function that takes a string -- used
       to report the progress of the write.
       
       'anchor' is a selection (i.e. instance of a subclass of
       chimera.selection.Selection) containing atoms/bonds that should
       be written out to the @SET section of the file as the rigid
       framework for flexible ligand docking.

       'hydNamingStyle' controls whether hydrogen names should be
       "Sybyl-like" (value: sybyl) or "PDB-like" (value: pdb)
       -- e.g.  HG21 vs. 1HG2.

       'multimodelHandling' controls whether multiple models will be
       combined into a single @MOLECULE section (value: combined) or
       each given its own section (value: individual).

       'skip' is a list of atoms to not output

       'resNum' controls whether residue sequence numbers are included
       in the substructure name.  Since Sybyl Mol2 files include them,
       this defaults to True.

       If 'gaffType' is True, outout GAFF atom types instead of Sybyl
       atom types.  'gaffFailError', if specified, is the type of error
       to throw (e.g. UserError) if there is no gaffType attribute for
       an atom, otherwise throw the standard AttributeError.

       If 'temporary' is True, don't enter the file name into the 
       "Recent Data Sources" list of the Rapid Access interface.
    """

    if isinstance(fileName, basestring):
        # open the given file name for writing
        from OpenSave import osOpen
        f = osOpen(fileName, "w")
        needClose = True
    else:
        f = fileName
        needClose = False

    sortFunc = serialSort = lambda a1, a2, ri={}: writeMol2Sort(a1, a2, resIndices=ri)

    if isinstance(models, chimera.Molecule):
        models = [models]
    elif isinstance(models, Selection):
        # create a fictitious jumbo model
        if isinstance(models, ItemizedSelection):
            sel = models
        else:
            sel = ItemizedSelection()
            sel.merge(models)
        sel.addImplied()
        class Jumbo:
            def __init__(self, sel):
                self.atoms = sel.atoms()
                self.residues = sel.residues()
                self.bonds = sel.bonds()
                self.name = "(selection)"
        models = [Jumbo(sel)]
        sortFunc = lambda a1, a2: cmp(a1.molecule.id, a2.molecule.id) \
            or cmp(a1.molecule.subid, a2.molecule.subid) \
            or serialSort(a1, a2)
        multimodelHandling = "individual"

    # transform...
    if relModel is None:
        xform = chimera.Xform.identity()
    else:
        xform = relModel.openState.xform
        xform.invert()

    # need to find amide moieties since Sybyl has an explicit amide type
    if status:
        status("Finding amides")
    from ChemGroup import findGroup
    amides = findGroup("amide", models)
    amideNs = dict.fromkeys([amide[2] for amide in amides])
    amideCNs = dict.fromkeys([amide[0] for amide in amides])
    amideCNs.update(amideNs)
    amideOs = dict.fromkeys([amide[1] for amide in amides])

    substructureNames = None
    if multimodelHandling == "combined":
        # create a fictitious jumbo model
        class Jumbo:
            def __init__(self, models):
                self.atoms = []
                self.residues = []
                self.bonds = []
                self.name = models[0].name + " (combined)"
                for m in models:
                    self.atoms.extend(m.atoms)
                    self.residues.extend(m.residues)
                    self.bonds.extend(m.bonds)
                # if combining single-residue models,
                # can be more informative to use model name
                # instead of residue type for substructure
                if len(models) == len(self.residues):
                    rtypes = [r.type for r in self.residues]
                    if len(set(rtypes)) < len(rtypes):
                        mnames = [m.name for m in models]
                        if len(set(mnames)) == len(mnames):
                            self.substructureNames = dict(
                                zip(self.residues, mnames))
        models = [Jumbo(models)]
        if hasattr(models[-1], 'substructureNames'):
            substructureNames = models[-1].substructureNames
            delattr(models[-1], 'substructureNames')
        sortFunc = lambda a1, a2: cmp(a1.molecule.id, a2.molecule.id) \
            or cmp(a1.molecule.subid, a2.molecule.subid) \
            or serialSort(a1, a2)

    # write out models
    for mol in models:
        if hasattr(mol, 'mol2comments'):
            for m2c in mol.mol2comments:
                print>>f, m2c
        if hasattr(mol, 'solventInfo' ):
            print>>f, mol.solventInfo

        # molecule section header
        print>>f, "%s" % MOLECULE_HEADER

        # molecule name
        print>>f, "%s" % mol.name

        ATOM_LIST = mol.atoms
        BOND_LIST = mol.bonds
        if skip:
            skip = set(skip)
            ATOM_LIST = [a for a in ATOM_LIST if a not in skip]
            BOND_LIST = [b for b in BOND_LIST
                    if b.atoms[0] not in skip
                    and b.atoms[1] not in skip]
        RES_LIST  = mol.residues

        # Chimera has an unusual internal order for its atoms, so
        # sort them by input order
        if status:
            status("Putting atoms in input order")
        ATOM_LIST.sort(sortFunc)

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
        if hasattr(mol, "mol2type"):
            mtype = mol.mol2type
        else:
            mtype = "SMALL"
            from chimera.resCode import nucleic3to1, protein3to1
            for r in mol.residues:
                if r.type in protein3to1:
                    mtype = "PROTEIN"
                    break
                if r.type in nucleic3to1:
                    mtype = "NUCLEIC_ACID"
                    break
        print>>f, mtype

        # indicate type of charge information
        if hasattr(mol, 'chargeModel'):
            print>>f, mol.chargeModel
        else:
            print>>f, "NO_CHARGES"

        if hasattr(mol, 'mol2comment'):
            print>>f, "\n%s" % mol.mol2comment
        else:
            print>>f, "\n"


        if status:
            status("writing atoms")
        # atom section header
        print>>f, "%s" % ATOM_HEADER

        # make a dictionary of residue indices so that we can do
        # quick look ups
        resIndices = {}
        for i, r in enumerate(RES_LIST):
            resIndices[r] = i+1
        for i, atom in enumerate(ATOM_LIST):
            # atom ID, starting from 1
            print>>f, "%7d" % (i+1),

            # atom name, possibly rearranged if it's a hydrogen
            if hydNamingStyle == "sybyl" \
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
            if gaffType:
                try:
                    atomType = atom.gaffType
                except AttributeError:
                    if not gaffFailError:
                        raise
                    raise gaffFailError("%s has no Amber/GAFF type assigned.\n"
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
                    atomType = chimera2sybyl[atom.idatmType]
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
            print>>f, "%5d" % resIndices[res],

            # substructure identifier and charge
            if hasattr(atom, 'charge') and atom.charge is not None:
                charge = atom.charge
            else:
                charge = 0.0
            if substructureNames:
                rname = substructureNames[res]
            elif resNum:
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
            elif resNum:
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
            if a.molecule.rootForAtom(a, True).atom.residue == res:
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
        triggers.activateTrigger('file save', (fileName, 'Mol2'))

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
