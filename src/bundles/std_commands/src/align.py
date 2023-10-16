# vim: set expandtab ts=4 sw=4:

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

from chimerax.core.errors import UserError
class IterationError(UserError):
    pass

def align(session, atoms, to_atoms = None, move = None, each = None,
          match_chain_ids = False, match_numbering = False, match_atom_names = False,
          sequence = None, cutoff_distance = None, report_matrix = False, log_info = True):
    """Move atoms to minimize RMSD with to_atoms.
    Returns matched atoms and matched to_atoms, matched atom rmsd, paired atom rmsd, and a
    chimerax.geometry.Place instance (i.e. the transform to place the match atoms onto
    the "to" atoms).  The matched atoms can be fewer than the paired atoms if cutoff distance
    is specified.  If "each" is not None then nothing is returned.

    If 'move' is 'structures', superimpose the models by changing the model positions.
    If it is 'atoms', 'residues', 'chains' or 'structure atoms', then atoms promoted extended
    to this level are moved.  If move is False move nothing or True move structures.
    If move is an Atoms collection then move only the specified atoms.

    If 'each' is "structure" then each structure of atoms is separately
    aligned to the to_atoms.  If 'each' is "chain" then each chain is
    aligned separately.  If 'each' is "coordset" then each coordinate set
    of the first set of atoms (which must belong to a single structure)
    is aligned. Default is that all atoms are aligned as one group.

    If 'match_chain_ids' is true then only atoms with matching chain identifiers are paired.
    Unpaired atoms or to_atoms are not used for alignment.

    If 'match_numbering' is true then only atoms with matching residue numbers are paired.
    It is assumed the atoms are in residue number order.  Unpaired atoms or to_atoms
    are not used.

    If 'match_atom_names' is true then only atoms with matching names are paired.
    Unpaired atoms or to_atoms are not used for alignment.

    If 'sequence' names a reference sequence then the previously calculated alignment
    of atoms and to_atoms to that reference sequence is used to pair the atoms.
    
    If 'report_matrix' is True, report the transformation matrix to
    the Reply Log.

    If 'log_info' is True, report RMSD and pruned atoms to the log.
    """
    if each == 'chain':
        groups = atoms.by_chain
        if move is None:
            move = 'chains'
        for s,cid,gatoms in groups:
            align(session, gatoms, to_atoms, move=move, match_chain_ids=match_chain_ids,
                  match_numbering=match_numbering, match_atom_names=match_atom_names,
                  sequence=sequence, report_matrix=report_matrix)
        return
    elif each == 'structure':
        groups = atoms.by_structure
        for s,gatoms in groups:
            align(session, gatoms, to_atoms, move=move, match_chain_ids=match_chain_ids,
                  match_numbering=match_numbering, match_atom_names=match_atom_names,
                  sequence=sequence, report_matrix=report_matrix)
        return
    elif each == 'coordset':
        us = atoms.unique_structures
        if len(us) != 1:
            raise UserError('Atoms must belong to a single structure to align each coordset, got %d structures'
                            % len(us))
        cset_mol = us[0]
        if move is None or move == 'structures':
            move = 'structure atoms'

    if move is None:
        move = 'structures'

    log = session.logger if log_info else None
    if sequence is None:
        patoms, pto_atoms = paired_atoms(atoms, to_atoms, match_chain_ids,
                                         match_numbering, match_atom_names)
        da, dra = len(atoms) - len(patoms), len(to_atoms) - len(pto_atoms)
        if log and (da > 0 or dra > 0):
            log.info('Pairing dropped %d atoms and %d reference atoms' % (da, dra))
    else:
        patoms, pto_atoms = sequence_alignment_pairing(atoms, to_atoms, sequence)

    npa, npta = len(patoms), len(pto_atoms)
    if npa != npta:
        raise UserError('Must align equal numbers of atoms, got %d and %d' % (npa, npta))
    elif npa == 0:
        raise UserError('No atoms paired for alignment')

    xyz_to = pto_atoms.scene_coords
    if each == 'coordset':
        cs = cset_mol.active_coordset_id
        for id in cset_mol.coordset_ids:
            cset_mol.active_coordset_id = id
            align_atoms(patoms, pto_atoms, xyz_to, cutoff_distance,
                        atoms, to_atoms, move, log, report_matrix)
        cset_mol.active_coordset_id = cs
        return

    return align_atoms(patoms, pto_atoms, xyz_to, cutoff_distance,
                       atoms, to_atoms, move, log, report_matrix)

def align_atoms(patoms, pto_atoms, xyz_to, cutoff_distance,
                atoms, to_atoms, move, log, report_matrix):

    xyz_from = patoms.scene_coords
    if cutoff_distance is None:
        from chimerax.geometry import align_points
        tf, rmsd = align_points(xyz_from, xyz_to)
        full_rmsd = rmsd
        matched_patoms, matched_pto_atoms = patoms, pto_atoms
        msg = 'RMSD between %d atom pairs is %.3f angstroms' % (len(patoms), rmsd)
    else:
        if cutoff_distance <= 0.0:
            raise UserError("Distance cutoff must be positive")
        tf, rmsd, indices = align_and_prune(xyz_from, xyz_to, cutoff_distance)
        np = len(indices)
        dxyz = tf*xyz_from - xyz_to
        d2 = (dxyz*dxyz).sum(axis=1)
        import math
        full_rmsd = math.sqrt(d2.sum() / len(d2))
        matched_patoms, matched_pto_atoms = patoms[indices], pto_atoms[indices]
        msg = 'RMSD between %d pruned atom pairs is %.3f angstroms;' \
            ' (across all %d pairs: %.3f)' % (np, rmsd, len(patoms), full_rmsd)

    if report_matrix and log:
        log.info(matrix_text(tf, atoms.structures[0]))

    if log:
        log.status(msg, log=True)

    move_atoms(atoms, to_atoms, tf, move)

    return matched_patoms, matched_pto_atoms, rmsd, full_rmsd, tf

def align_and_prune(xyz, ref_xyz, cutoff_distance, indices = None):

    import numpy
    if indices is None:
        indices = numpy.arange(len(xyz))
    axyz, ref_axyz = xyz[indices], ref_xyz[indices]
    from chimerax.geometry import align_points
    p, rms = align_points(axyz, ref_axyz)
    dxyz = p*axyz - ref_axyz
    d2 = (dxyz*dxyz).sum(axis=1)
    cutoff2 = cutoff_distance * cutoff_distance
    i = d2.argsort()
    if d2[i[-1]] <= cutoff2:
        return p, rms, indices

    # cull 10% or...
    index1 = int(len(d2) * 0.9)
    # cull half the long pairings
    index2 = int(((d2 <= cutoff2).sum() + len(d2)) / 2)
    # whichever is fewer
    index = max(index1, index2)
    survivors = indices[i[:index]]

    if len(survivors) < 3:
        raise IterationError("Alignment failed;"
            " pruning distances > %g left less than 3 atom pairs" % cutoff_distance)
    return align_and_prune(xyz, ref_xyz, cutoff_distance, survivors)

def paired_atoms(atoms, to_atoms, match_chain_ids, match_numbering, match_atom_names):
    # TODO: return summary string of all dropped atoms.
    if match_chain_ids:
        pat = pair_chains(atoms, to_atoms)
    elif match_numbering:
        # Pair chains in order of matching sequence numbers but not chain ids.
        ca, cta = atoms.by_chain, to_atoms.by_chain
        n = min(len(ca), len(cta))
        # TODO: Warn if unequal number of sequences.
        pat = [(ca[i][2], cta[i][2]) for i in range(n)]
    else:
        pat = [(atoms,to_atoms)]

    if match_numbering:
        pat = sum([pair_sequence_numbers(a, ta) for a,ta in pat], [])

    if match_atom_names:
        pat = sum([pair_atom_names(a, ta) for a,ta in pat], [])

    for pa, pta in pat:
        if len(pa) != len(pta):
            msg = 'Unequal number of atoms to pair, %d and %d' % (len(pa), len(pta))
            if match_chain_ids:
                msg += ', chain %s' % pa[0].chain_id
            if match_numbering:
                msg += ', residue %d' % pa[0].residue.number
            raise UserError(msg)

    from chimerax.atomic import Atoms, concatenate
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

def matrix_text(tf, m):

    mp = m.position
    mtf = mp.inverse() * tf * mp
    msg = ('Alignment matrix in structure %s coordinates\n%s' % (m.name, mtf.description()))
    return msg

def move_atoms(atoms, to_atoms, tf, move):

    if move == 'structures' or move is True:
        for m in atoms.unique_structures:
            m.scene_position = tf * m.scene_position
    else:
        from chimerax.atomic import Atoms
        if move == 'atoms':
            matoms = atoms
        elif move == 'residues':
            matoms = atoms.unique_residues.atoms
        elif move == 'chains':
            matoms = extend_to_chains(atoms)
        elif move == 'structure atoms':
            from chimerax.atomic import concatenate, Atoms
            matoms = concatenate([m.atoms for m in atoms.unique_structures], Atoms)
        elif isinstance(move, Atoms):
            matoms = move
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
    from chimerax.atomic import concatenate, Atoms
    return concatenate(catoms, Atoms)
        
def register_command(logger):

    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, FloatArg, IntArg
    from chimerax.atomic import OrderedAtomsArg
    desc = CmdDesc(required = [('atoms', OrderedAtomsArg)],
                   keyword = [('to_atoms', OrderedAtomsArg),
                              ('move', EnumOf(('atoms', 'residues', 'chains', 'structures',
                                               'structure atoms', 'nothing'))),
                              ('each', EnumOf(('chain', 'structure', 'coordset'))),
                              ('match_chain_ids', BoolArg),
                              ('match_numbering', BoolArg),
                              ('match_atom_names', BoolArg),
                              ('cutoff_distance', FloatArg),
                              ('report_matrix', BoolArg)],
                   required_arguments = ['to_atoms'],
                   synopsis = 'Align one set of atoms to another')
    register('align', desc, align, logger=logger)
