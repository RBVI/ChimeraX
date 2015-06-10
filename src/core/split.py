# -----------------------------------------------------------------------------
# Command to split molecules so that each chain is in a separate molecule.
#
def split_molecules(session, molecules = None, chains = None, ligands = False, connected = False, atoms = None):

    if molecules is None:
        from . import structure
        molecules = structure.all_atomic_structures(session)

    if chains is None and not ligands and not connected and atoms is None:
        chains = True

    slist = []
    olist = []
    log = session.logger
    models = session.models
    from .models import Model
    for m in molecules:
        clist = split_molecule(m, chains, ligands, connected, atoms)
        if clist:
            parent = Model(m.name)
            parent.id = m.id
            for i, c in enumerate(clist):
                c.id = parent.id + (i+1,)
                c.position = m.position
            parent.add(clist)
            slist.append(m)
            olist.append(parent)
            msg = 'Split %s (#%s) into %d models' % (m.name, m.id_string(), len(clist))
        else:
            msg = 'Did not split %s, has only one piece' % m.name
        log.status(msg)
        log.info(msg)

    models.close(slist)
    models.add(olist)
    
# -----------------------------------------------------------------------------
#
def split_molecule(m, chains, ligands, connected, atoms):

    pieces = [(m.name, m.atoms)]
    if chains:
        pieces= split_pieces(pieces, split_by_chain)
    if ligands:
        pieces = split_pieces(pieces, split_by_ligand)
    if connected:
        pieces = split_pieces(pieces, split_connected)
    if atoms:
        pieces = split_pieces(pieces, lambda a,atoms=atoms: split_atoms(a,atoms))
    
    if len(pieces) == 1:
        return []
    
    mlist = [molecule_from_atoms(m, atoms, name) for name, atoms in pieces]
    return mlist
    
# -----------------------------------------------------------------------------
#
def split_pieces(pieces, split_function):

    plist = []
    for name, atoms in pieces:
        splist = split_function(atoms)
        if len(splist) == 1:
            plist.append((name,atoms))
        else:
            plist.extend((('%s %s' % (name,n) if n else name), a) for n,a in splist)
    return plist
    
# -----------------------------------------------------------------------------
#
def split_by_chain(atoms):

    ca = list(atoms_by_chain(atoms).items())
    ca.sort()
    return ca
    
# -----------------------------------------------------------------------------
#
def atoms_by_chain(atoms):

    ct = {}
    for a in atoms:
        cid = a.residue.chain_id
        if cid in ct:
            ct[cid].append(a)
        else:
            ct[cid] = [a]
    return ct
    
# -----------------------------------------------------------------------------
#
def split_by_ligand(atoms):

    latoms = []
    oatoms = []
# TODO: identify ligands
    # for a in atoms:
    #     if a.surfaceCategory == 'ligand':
    #         latoms.append(a)
    #     else:
    #         oatoms.append(a)
    pieces = [('', oatoms)] if oatoms else []
    if latoms:
        for n,a in split_pieces([('', latoms)], split_connected):
            pieces.append((a[0].residue.type, a))
    return pieces
    
# -----------------------------------------------------------------------------
#
def split_connected(atoms):

    aset = set(atoms)
    reached = {}        # Map atom to tuple of connected atoms
    for a in atoms:
        j = set([a])
        for b in a.bonds:
            a2 = b.other_atom(a)
            if a2 in aset and a2 in reached:
                j.update(reached[a2])
        j = tuple(j)
        for a3 in j:
            reached[a3] = j
    cats = list(set(reached.values()))
    cats.sort(key = lambda cat: len(cat))
    cats.reverse()                              # Number largest to smallest
    pieces = ([('', cats)] if len(cats) == 1
              else [('%d' % (i+1,), cat) for i,cat in enumerate(cats)])
    return pieces
    
# -----------------------------------------------------------------------------
#
def split_atoms(atoms, asubsets):

    # Eliminate subset atoms not in atoms
    aset = set(atoms)
    asubsets = [tuple(a for a in asub if a in aset) for asub in asubsets]

    # Remove empty subsets
    asubsets = [asub for asub in asubsets if len(asub) > 0]

    # Find atoms not in any subset
    for asub in asubsets:
        for a in asub:
            aset.discard(a)
    a0 = [a for a in atoms if a in aset]

    # Return groups of atoms
    if len(a0) == len(atoms):
        pieces = [('',atoms)]
    elif len(a0) == 0 and len(asubsets) == 1:
        pieces = [('',atoms)]
    else:
        alists = (asubsets + [a0]) if a0 else asubsets
        pieces = [(str(i+1),a) for i,a in enumerate(alists)]

    return pieces

# -----------------------------------------------------------------------------
#
def molecule_from_atoms(m, atoms, name = None):

    from .structure import AtomicStructure
    cm = AtomicStructure(name or m.name)
#    cm.color = m.color
    cm.display = m.display
#    cm.lineWidth = m.lineWidth
#    cm.pointSize = m.pointSize
#    cm.ballScale = m.ballScale

#    cm.pdbVersion = m.pdbVersion
#    if hasattr(m, 'pdbHeaders'):
#        cm.setAllPDBHeaders(m.pdbHeaders)
#    if hasattr(m, 'mmCIFHeaders'):
#        cm.mmCIFHeaders = m.mmCIFHeaders

    rmap = {}
    rlist = atom_residues(atoms)
    rorder = dict((r,i) for i,r in enumerate(m.residues))
    rlist.sort(key = lambda r: rorder[r])
    for r in rlist:
        cr = cm.new_residue(r.name, r.chain_id, r.number)
#        cr.isHet = r.isHet
        cr.is_helix = r.is_helix
        cr.is_sheet = r.is_sheet
        cr.ribbon_color = r.ribbon_color
#        cr.ribbonStyle = r.ribbonStyle
#        cr.ribbonDrawMode = r.ribbonDrawMode
        cr.ribbon_display = r.ribbon_display
        rmap[r] = cr

    amap = {}
    for a in atoms:
        ca = cm.new_atom(a.name, a.element_name)
        ca.coord = a.coord
#        ca.altLoc = a.altLoc
        ca.color = a.color
        ca.draw_mode = a.draw_mode
        ca.display = a.display
        ca.bfactor = a.bfactor
        amap[a] = ca
        cr = rmap[a.residue]
        cr.add_atom(ca)

    for b in atom_bonds(atoms):
        a1, a2 = b.atoms
        cb = cm.new_bond(amap[a1], amap[a2])
        cb.color = b.color
        cb.radius = b.radius
#        cb.drawMode = b.drawMode
        cb.display = b.display
        cb.halfbond = b.halfbond

    for name, pbg in m.pbg_map.items():
        apairs = []
        for pb in pbg.pseudoBonds:
            a1, a2 = pb.atoms
            if a1 not in amap or a2 not in amap:
                continue
            apairs.append((amap[a1], amap[a2]))
        if apairs:
            cpbg = cm.pseudobond_group(name)
            for a1, a2 in apairs:
                cpbg.new_pseudobond(a1,a2)

    cm.new_atoms()

    return cm

# -----------------------------------------------------------------------------
#
def atom_residues(atoms):

    rt = {}
    for a in atoms:
        rt[a.residue] = 1
    rlist = list(rt.keys())
    return rlist

# -----------------------------------------------------------------------------
# Bonds with both ends in given atom set.
#
def atom_bonds(atoms):

    at = {}
    for a in atoms:
        at[a] = 1
    bt = {}
    for a in atoms:
        for b in a.bonds:
            if not b in bt:
                a1, a2 = b.atoms
                if a1 in at and a2 in at:
                    bt[b] = 1
    blist = bt.keys()
    return blist

# -----------------------------------------------------------------------------
#
def register_split_command():

    from . import cli, atomspec, color, structure
    desc = cli.CmdDesc(
        optional = [('molecules', structure.AtomicStructuresArg)],
        keyword = [('chains', cli.NoArg),
                   ('ligands', cli.NoArg),
                   ('connected', cli.NoArg),
                   ('atoms', structure.AtomsArg)],
        synopsis = 'split molecule into multiple molecules'
        )
    # TODO: Allow repeating atoms keyword.

    cli.register('split', desc, split_molecules)
