# vim: set expandtab ts=4 sw=4:

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

def similar_structures_ligands(session, rmsd_cutoff = 3.0, alignment_range = 5.0, minimum_paired = 0.5,
                               combine = True, from_set = None, of_structures = None, warn = True):
    from .simstruct import similar_structure_results
    results = similar_structure_results(session, from_set)
    hits = results.named_hits(of_structures)

    query_chain = results.query_chain
    if query_chain is None:
        from chimerax.core.errors import UserError
        raise UserError('Cannot position Foldseek ligands without query structure')

    if warn and session.ui.is_gui:
        message = f'This will fetch {len(hits)} PDB structures and align their ligands to the query structure.  It may take several minutes to fetch those structures during which ChimeraX will be frozen.  Do you want to proceed?'
        from chimerax.ui.ask import ask
        answer = ask(session, message, title='Fetch similar structure ligands')
        if answer == 'no':
            return False

    keep_structs = []
    nlighits = 0
    from time import time
    t0 = time()
    for hnum, hit in enumerate(hits):
        structures = results.open_hit(session, hit, align = False,
                                      in_file_history = False, log = False)

        hname = hit['database_full_id']
        telapse = _minutes_and_seconds_string(time() - t0)
        session.logger.status(f'Finding ligands in {hname} ({hnum+1} of {len(hits)}, time {telapse})')

        found_lig = False
        for si, structure in enumerate(structures):
            res = structure.residues
            from chimerax.atomic import Residue
            ligres = res[res.polymer_types == Residue.PT_NONE]
            aares = res[res.polymer_types == Residue.PT_AMINO]
            keeplig = []
            if ligres:
                from .simstruct import hit_and_query_residue_pairs
                rmap = {hr:qr for hr,qr in hit_and_query_residue_pairs(structure, query_chain, hit)}
                for lr in ligres:
                    cres = _find_close_residues(lr, aares, alignment_range)
                    if len(cres) >= 3:
                        pcres, qres = _paired_residues(rmap, cres)
                        if len(qres) >= 3 and len(qres) >= minimum_paired * len(cres):
                            from .simstruct import alignment_transform
                            p, rms, npairs = alignment_transform(pcres, qres)
                            if rms <= rmsd_cutoff:
                                keeplig.append(lr)
                                # Remove other alt locs so we don't have to move them.
                                lr.clean_alt_locs()
                                atoms = lr.atoms
                                atoms.coords = p.transform_points(atoms.coords)
                                atoms.displays = True
            if keeplig:
                _delete_extra_residues(res, keeplig)
                keep_structs.append(structure)
                if len(structures) > 1:
                    structure.ensemble_id = si+1
                found_lig = True
                # TODO: Slows down a lot when many structures open in session
                #   Takes 8 minutes instead of 45 minutes if I remove from session for 8jnb 845 hits.
                session.models.remove([structure])
            else:
                session.models.close([structure])
        if found_lig:
            nlighits += 1

    session.models.add(keep_structs)
    nlig = 0
    lignames = set()
    for structure in keep_structs:
        nlig += structure.num_residues
        lignames.update(structure.residues.names)
    lignames = ', '.join(sorted(lignames))
    session.logger.status(f'Found {nlig} ligands in {nlighits} hits: {lignames}', log = True)

    if combine and keep_structs:
        _include_pdb_id_in_chain_ids(keep_structs)
        cmodel =_combine_structures(session, keep_structs)
        return cmodel

    return keep_structs

def _minutes_and_seconds_string(tsec):
    tmin = int(tsec/60)
    ts = int(tsec - tmin*60)
    return '%d:%02d' % (tmin, ts)

def _find_close_residues(residue, residues, distance):
    rxyz = residue.atoms.coords
    aatoms = residues.atoms
    axyz = aatoms.coords
    from chimerax.geometry import find_close_points
    ri, ai = find_close_points(rxyz, axyz, distance)
    close_res = aatoms[ai].residues.unique()
    return close_res

def _paired_residues(rmap, residues):
    r1 = []
    r2 = []
    for r in residues:
        if r in rmap:
            r1.append(r)
            r2.append(rmap[r])
    from chimerax.atomic import Residues
    return Residues(r1), Residues(r2)

def _delete_extra_residues(residues, keep_residues):
    keep = set(keep_residues)
    del_res = [r for r in residues if r not in keep]
    if del_res:
        from chimerax.atomic import Residues
        Residues(del_res).delete()

def _include_pdb_id_in_chain_ids(structures):
    # If hits for different chains of one PDB exist use longer chain prefixes.
    # For example if 7mrj_A and 7mrj_B are hits and ligand /A:114 is found in both
    # then use chain ids 7mrj_A_A and 7mrj_B_A so the combine command does not get
    # clashing chain ids and residue numbers
    pdb_ids = set()
    multi_hit = set()
    for structure in structures:
        pdb_id = structure.name.split('_')[0]
        if pdb_id in pdb_ids:
            multi_hit.add(pdb_id)
        pdb_ids.add(pdb_id)

    for structure in structures:
        cids = tuple(set(structure.residues.chain_ids))
        chain_ids = ','.join(cids)
        pdb_id = structure.name.split('_')[0]
        prefix = (structure.name if pdb_id in multi_hit else pdb_id) + '_'
        if hasattr(structure, 'ensemble_id'):
            prefix += str(structure.ensemble_id) + '_'
        new_chain_ids = ','.join(prefix + cid for cid in cids)
        cmd = f'changechains #{structure.id_string} {chain_ids} {new_chain_ids} log false'
        from chimerax.core.commands import run
        run(structure.session, cmd, log = False)
    
def _combine_structures(session, structures):
    from chimerax.core.commands import concise_model_spec, run
    mspec = concise_model_spec(session, structures)
    cmd = f'combine {mspec} close true retainIds true name "similar structure ligands"'
    cmodel = run(session, cmd, log = False)
    return cmodel

def register_similar_structures_ligands_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, BoolArg, StringArg
    desc = CmdDesc(
        keyword = [('rmsd_cutoff', FloatArg),
                   ('alignment_range', FloatArg),
                   ('minimum_paired', FloatArg),
                   ('combine', BoolArg),
                   ('from_set', StringArg),
                   ('of_structures', StringArg),
                   ('warn', BoolArg),
                   ],
        synopsis = 'Find ligands in Foldseek hits and align to query.'
    )
    register('similarstructures ligands', desc, similar_structures_ligands, logger=logger)
