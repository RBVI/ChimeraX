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

def align(session, atoms, to_atoms = None, move = None, each = None,
          match_chain_ids = False, match_sequence_numbers = False, match_atom_names = False,
          sequence = None, cutoff_distance = None, iterations = 20, report_matrix=False):
    """Move atoms to minimize RMSD with to_atoms.

    If 'move' is 'molecules', superimpose the models.  If it is 'atoms',
    'residues' or 'chains' move just those atoms.  If move is False
    move nothing or True move molecules.  If move is a tuple, list or
    set of atoms then move those.

    If 'each' is "molecule" then each molecule of atoms is separately
    aligned to the to_atoms.  If 'each' is "chain" then each chain is
    aligned separately.  Default is that all atoms are aligned as one group.

    If 'match_chain_ids' is true then only atoms with matching chain identifiers are paired.
    Unpaired atoms or to_atoms are not used for alignment.

    If 'match_sequence_numbers' is true then only atoms with matching residue numbers are paired.
    It is assumed the atoms are in residue number order.  Unpaired atoms or to_atoms
    are not used.

    If 'match_atom_names' is true then only atoms with matching names are paired.
    Unpaired atoms or to_atoms are not used for alignment.

    If 'sequence' names a reference sequence then the previously calculated alignment
    of atoms and to_atoms to that reference sequence is used to pair the atoms.
    
    If 'report_matrix' is True, report the transformation matrix to
    the Reply Log.
    """
    if to_atoms is None:
        from . import AnnotationError
        raise AnnotationError('Require "to" keyword: align #1 to #2')

    if each == 'chain':
        groups = atoms.by_chain
        if move is None:
            move = 'chains'
        for s,cid,gatoms in groups:
            align(session, gatoms, to_atoms, move=move, match_chain_ids=match_chain_ids,
                  match_sequence_numbers=match_sequence_numbers, match_atom_names=match_atom_names,
                  sequence=sequence, report_matrix=report_matrix)
        return
    elif each == 'molecule':
        groups = atoms.by_structure
        for s,gatoms in groups:
            align(session, gatoms, to_atoms, move=move, match_chain_ids=match_chain_ids,
                  match_sequence_numbers=match_sequence_numbers, match_atom_names=match_atom_names,
                  sequence=sequence, report_matrix=report_matrix)
        return

    log = session.logger
    if sequence is None:
        patoms, pto_atoms = paired_atoms(atoms, to_atoms, match_chain_ids,
                                         match_sequence_numbers, match_atom_names)
        da, dra = len(atoms) - len(patoms), len(to_atoms) - len(pto_atoms)
        if da > 0 or dra > 0:
            log.info('Pairing dropped %d atoms and %d reference atoms' % (da, dra))
    else:
        patoms, pto_atoms = sequence_alignment_pairing(atoms, to_atoms, sequence)

    npa, npta = len(patoms), len(pto_atoms)
    if npa != npta:
        from ..errors import UserError
        raise UserError('Must align equal numbers of atoms, got %d and %d' % (npa, npta))
    elif npa == 0:
        from ..errors import UserError
        raise UserError('No atoms paired for alignment')
        
    if cutoff_distance is None:
        from ..geometry import align_points
        tf, rmsd = align_points(patoms.scene_coords, pto_atoms.scene_coords)
        np = npa
    else:
        tf, rmsd, mask = align_and_prune(patoms.scene_coords, pto_atoms.scene_coords,
                                         cutoff_distance, iterations)
        if tf is None:
            msg = ('Alignment failed, pruning %d distances > %g left less than 3 pairs'
                   % (npa, cutoff_distance))
            log.status(msg)
            log.info(msg)
            return
        np = mask.sum()

    if report_matrix:
        log.info(matrix_text(tf, atoms, to_atoms))

    msg = 'RMSD between %d atom pairs is %.3f Angstroms' % (np, rmsd)
    log.status(msg)
    log.info(msg)

    if move is None:
        move = 'molecules'

    move_atoms(atoms, to_atoms, tf, move)

    return tf, rmsd

def align_and_prune(xyz, ref_xyz, cutoff_distance, iterations, mask = None):

    axyz, ref_axyz = (xyz, ref_xyz) if mask is None else (xyz[mask,:], ref_xyz[mask,:])
    from ..geometry import align_points
    p, rms = align_points(axyz, ref_axyz)
    dxyz = p*xyz - ref_xyz
    d2 = (dxyz*dxyz).sum(axis=1)
    c = (d2 <= cutoff_distance*cutoff_distance)
    nc = c.sum()
    if nc == len(xyz) or (not mask is None and (c == mask).all()):
        return p, rms, c
    if iterations <= 1:
        print ('prune fail', nc, iterations, rms, len(xyz), len(axyz))
        return None, None, None
    if nc < 3 and len(c) > 3:
        # TODO: This method of avoiding overpruning is not well thought out.
        dm2 = cutoff_distance*cutoff_distance
        while True:
            dm2 *= 1.5
            c = (d2 <= dm2)
            if c.sum() > len(c)/2:
                break
    return align_and_prune(xyz, ref_xyz, cutoff_distance, iterations-1, c)

def paired_atoms(atoms, to_atoms, match_chain_ids, match_sequence_numbers, match_atom_names):
    # TODO: return summary string of all dropped atoms.
    if match_chain_ids:
        pat = pair_chains(atoms, to_atoms)
    elif match_sequence_numbers:
        # Pair chains in order of matching sequence numbers but not chain ids.
        ca, cta = atoms.by_chain, to_atoms.by_chain
        n = min(len(ca), len(cta))
        # TODO: Warn if unequal number of sequences.
        pat = [(ca[i][2], cta[i][2]) for i in range(n)]
    else:
        pat = [(atoms,to_atoms)]

    if match_sequence_numbers:
        pat = sum([pair_sequence_numbers(a, ta) for a,ta in pat], [])

    if match_atom_names:
        pat = sum([pair_atom_names(a, ta) for a,ta in pat], [])

    for pa, pta in pat:
        if len(pa) != len(pta):
            msg = 'Unequal number of atoms to pair, %d and %d' % (len(pa), len(pta))
            if match_chain_ids:
                msg += ', chain %s' % pa[0].chain_id
            if match_sequence_numbers:
                msg += ', residue %d' % pa[0].residue.number
            from ..errors import UserError
            raise UserError(msg)

    from ..atomic import Atoms, concatenate
    pas = concatenate([pa[0] for pa in pat], Atoms)
    ptas = concatenate([pa[1] for pa in pat], Atoms)

    return pas, ptas

def pair_chains(atoms, to_atoms):
    ca = value_index_map(atoms.chain_ids)
    cta = value_index_map(to_atoms.chain_ids)
    cp = [(atoms.filter(i1), to_atoms.filter(i2)) for i1, i2 in pair_items(ca, cta)]
    return cp

def pair_sequence_numbers(atoms, to_atoms):
    sa = value_index_map(atoms.residues.numbers)
    sta = value_index_map(to_atoms.residues.numbers)
    sp = [(atoms.filter(i1), to_atoms.filter(i2)) for i1, i2 in pair_items(sa, sta)]
    return sp

def pair_atom_names(atoms, to_atoms):
    na = value_index_map(atoms.names)
    nta = value_index_map(to_atoms.names)
    ap = [(atoms.filter(i1), to_atoms.filter(i2)) for i1, i2 in pair_items(na, nta)]
    return ap

def value_index_map(array):
    vmap = {}
    for i,v in enumerate(array):
        if v in vmap:
            vmap[v].append(i)
        else:
            vmap[v] = [i]
    return vmap

def pair_items(map1, map2):
    pairs = []
    for k, v1 in map1.items():
        if k in map2:
            v2 = map2[k]
            n1, n2 = len(v1), len(v2)
            if n1 == n2:
                pairs.append((v1, v2))
            else:
                n = min(n1, n2)
                pairs.append((v1[:n], v2[:n]))
    return pairs

# Pair atoms with same name and same alignment sequence number.
def sequence_alignment_pairing(atoms, to_atoms, seq_name):
    snums = atoms.sequence_numbers(seq_name)
    anames = atoms.atom_names()
    aind = dict(((snums[i],anames[i]),i) for i in range(atoms.count()) if snums[i] > 0)
    rsnums = to_atoms.sequence_numbers(seq_name)
    ranames = to_atoms.atom_names()
    ai = []
    rai = []
    for ri in range(to_atoms.count()):
        i = aind.get((rsnums[ri],ranames[ri]), None)
        if not i is None:
            ai.append(i)
            rai.append(ri)
    pa, pra = atoms.subset(ai), to_atoms.subset(rai)
    return pa, pra

def matrix_text(tf, atoms, to_atoms):

    m = atoms.structures[0]
    mp = m.position
    mtf = mp.inverse() * tf * mp
    msg = ('Alignment matrix in molecule %s coordinates\n%s' % (m.name, mtf.description()))
    return msg

def move_atoms(atoms, to_atoms, tf, move):

    if move == 'molecules' or move is True:
        for m in atoms.unique_structures:
            m.scene_position = tf * m.scene_position
    else:
        if move == 'atoms':
            matoms = atoms
        elif move == 'residues':
            matoms = atoms.unique_residues.atoms
        elif move == 'chains':
            matoms = extend_to_chains(atoms)
        else:
            return	# Move nothing

        matoms.scene_coords = tf * matoms.scene_coords

def extend_to_chains(atoms):
    '''Return atoms extended to all atoms in same chains (ie same chain id / structure).'''
    catoms = []
    from numpy import in1d
    for s, a in atoms.by_structure:
        satoms = s.atoms
        catoms.append(satoms.filter(in1d(satoms.chain_ids, a.unique_chain_ids)))
    from ..atomic import concatenate, Atoms
    return concatenate(catoms, Atoms)
        
def register_command(session):

    from . import CmdDesc, register, AtomsArg, EnumOf, NoArg, FloatArg, IntArg
    desc = CmdDesc(required = [('atoms', AtomsArg)],
                   keyword = [('to_atoms', AtomsArg),
                              ('move', EnumOf(('atoms', 'residues', 'chains', 'molecules', 'nothing'))),
                              ('each', EnumOf(('chain', 'molecule'))),
                              ('match_chain_ids', NoArg),
                              ('match_sequence_numbers', NoArg),
                              ('match_atom_names', NoArg),
                              ('cutoff_distance', FloatArg),
                              ('iterations', IntArg),
                              ('report_matrix', NoArg)],
                   synopsis = 'Align one set of atoms to another')
    register('align', desc, align)
