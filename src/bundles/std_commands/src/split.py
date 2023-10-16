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

# -----------------------------------------------------------------------------
# Command to split molecules so that each chain is in a separate molecule.
#
def split(session, structures = None, chains = None, ligands = False, connected = False, atoms = None):
    '''
    Partition atoms into separate structures. If only the first argument is given then those
    structures are split into a separate structure for each chain.  If chains, ligands, connected,
    or atoms keywords are given then additional partitioning into smaller subsets is performed.

    Parameters
    ----------
    Structures : Structures or None
      Structures to split.  If no structures specified then all are used.
    chains : bool
      Split each chain into into a separate atomic structure.
    ligands : bool
      Split each connected ligand into a separate atomic structure.
    connected : bool
      Split each connected subset of atoms into a separate atomic structure.
    atoms : list of Atoms
      Split specified atoms into separate atomic structures.  This option
      can be specified multiple times.
    '''
    if structures is None:
        from chimerax import atomic
        structures = atomic.all_atomic_structures(session)
        
    if chains is None and not ligands and not connected and atoms is None:
        chains = True

    if atoms:
        check_for_overlapping_atoms(atoms)
        
    slist = []
    olist = []
    log = session.logger
    from chimerax.core.models import Model
    for m in structures:
        clist = split_molecule(m, chains, ligands, connected, atoms)
        if clist:
            parent = Model(m.name, session)
            parent.id = m.id
            for i, c in enumerate(clist):
                c.id = parent.id + (i+1,)
                c.position = m.position
            parent.add(clist)
            slist.append(m)
            olist.append(parent)
            msg = 'Split %s (#%s) into %d models' % (m.name, m.id_string, len(clist))
        else:
            msg = 'Did not split %s, has only one piece' % m.name
        log.status(msg)
        log.info(msg)

    models = session.models
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
    
    mlist = [molecule_from_atoms(m, patoms, name) for name, patoms in pieces]
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

    ca = [(cid, atoms) for m, cid, atoms in atoms.by_chain]
    ca.sort()
    return ca
    
# -----------------------------------------------------------------------------
#
def split_by_ligand(atoms):

    ligmask = (atoms.structure_categories == 'ligand')
    latoms = atoms.filter(ligmask)
    oatoms = atoms.filter(~ligmask)
    pieces = [('', oatoms)] if len(oatoms) else []
    if len(latoms) > 0:
        for n,a in split_pieces([('', latoms)], split_connected):
            pieces.append((a[0].residue.name, a))
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
    from chimerax.atomic import Atoms
    pieces = ([('', Atoms(cats[0]))] if len(cats) == 1
              else [('%d' % (i+1,), Atoms(cat)) for i,cat in enumerate(cats)])
    return pieces
    
# -----------------------------------------------------------------------------
#
def split_atoms(atoms, asubsets):

    # Eliminate subset atoms not in atoms
    asubsets = [asub.intersect(atoms) for asub in asubsets]

    # Remove empty subsets
    asubsets = [asub for asub in asubsets if len(asub) > 0]

    # Find atoms not in any subset
    from chimerax.atomic import concatenate, Atoms
    a0 = atoms.subtract(concatenate(asubsets, Atoms))

    # Return groups of atoms
    if len(a0) == len(atoms):
        pieces = [('',atoms)]
    elif len(a0) == 0 and len(asubsets) == 1:
        pieces = [('',atoms)]
    else:
        alists = (asubsets + [a0]) if len(a0) > 0 else asubsets
        pieces = [(str(i+1),a) for i,a in enumerate(alists)]

    return pieces

# -----------------------------------------------------------------------------
#
def molecule_from_atoms(m, atoms, name = None):

    from chimerax.atomic import AtomicStructure, Structure
    structure_class = AtomicStructure if isinstance(m, AtomicStructure) else Structure
    cm = structure_class(m.session, name = (name or m.name), auto_style = False)
    cm.ss_assigned = True
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
        cid = r.chain_id
        cr = cm.new_residue(r.name, cid, r.number)
#        cr.isHet = r.isHet
        cr.ss_type = r.ss_type
        cr.ribbon_color = r.ribbon_color
#        cr.ribbonStyle = r.ribbonStyle
#        cr.ribbonDrawMode = r.ribbonDrawMode
        cr.ribbon_display = r.ribbon_display
        rmap[r] = cr

    amap = {}
    for a in atoms:
        ca = cm.new_atom(a.name, a.element.name)
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
        cpbgs = {}
        for pb in pbg.pseudobonds:
            a1, a2 = pb.atoms
            if a1 not in amap or a2 not in amap:
                continue
            cpbg = cpbgs.get(name)
            if cpbg is None:
                cpbgs[name] = cpbg = cm.pseudobond_group(name, create_type = pbg.group_type)
            cpb = cpbg.new_pseudobond(amap[a1],amap[a2])
            cpb.display = pb.display
            cpb.color = pb.color
            cpb.radius = pb.radius
            cpb.halfbond = pb.halfbond

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
def check_for_overlapping_atoms(atom_subsets):
    if atom_subsets is None or len(atom_subsets) < 2:
        return
    from chimerax.atomic import concatenate
    atoms = concatenate(atom_subsets, remove_duplicates = True)
    na = sum(len(a) for a in atom_subsets)
    if len(atoms) == na:
        return

    from chimerax.core.errors import UserError
    have_spec = [a for a in atom_subsets if isinstance(getattr(a, 'spec', None), str)]
    if len(have_spec) < len(atom_subsets):
        raise UserError('The split command requires non-overlapping atoms, '
                        '%d atoms are overlapping' % (na - len(atoms)))

    olap = []
    for i,a in enumerate(atom_subsets):
        for a2 in atom_subsets[i+1:]:
            ia = a.intersect(a2)
            if len(ia) > 0:
                olap.append(f'"{a.spec}" and "{a2.spec}" have {len(ia)} atoms in common')
    raise UserError('The split command requires non-overlapping atoms, ' + ', '.join(olap))
        
# -----------------------------------------------------------------------------
#
def register_command(logger):

    from chimerax.core.commands import CmdDesc, register, NoArg, RepeatOf
    from chimerax.atomic import AtomsArg, StructuresArg
    desc = CmdDesc(
        optional = [('structures', StructuresArg)],
        keyword = [('chains', NoArg),
                   ('ligands', NoArg),
                   ('connected', NoArg),
                   ('atoms', RepeatOf(AtomsArg))],
        synopsis = 'split molecule into multiple molecules'
        )

    register('split', desc, split, logger=logger)
