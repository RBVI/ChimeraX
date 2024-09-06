# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
        
def open_hit(session, hit, query_chain, trim = True, align = True, alignment_cutoff_distance = 2.0,
             in_file_history = True, log = True):
    af_frag = hit.get('alphafold_fragment')
    if af_frag is not None and af_frag != 'F1':
        session.logger.warning(f'Foldseek AlphaFold database hit {hit["alphafold_id"]} was predicted in fragments and ChimeraX is not able to fetch the fragment structures')
        return

    db_id = hit.get('database_id')
    if in_file_history:
        from_db = 'from alphafold' if hit['database'].startswith('afdb') else ''
        open_cmd = f'open {db_id} {from_db}'
        if not log:
            open_cmd += ' logInfo false'
        from chimerax.core.commands import run
        structures = run(session, open_cmd, log = log)
    else:
        kw = {'from_database': 'alphafold'} if hit['database'].startswith('afdb') else {}
        structures, status = session.open_command.open_data(db_id, log_info = log, **kw)
        session.models.add(structures)

    name = hit.get('database_full_id')
    # Can get multiple structures such as NMR ensembles from PDB.
    stats = []
    for structure in structures:
        structure.name = name
        _remember_sequence_alignment(structure, query_chain, hit)
        _check_structure_sequence(structure, hit)
        _trim_structure(structure, hit, trim, log = log)
        if structure.deleted:
            continue  # No residues aligned and every residue trimmed
        _show_ribbons(structure)

        if query_chain is not None and align:
            # Align the model to the query structure using Foldseek alignment.
            # Foldseek server does not return transform by default, so compute from sequence alignment.
            hit_chain = _hit_chain(structure, hit)
            if hit_chain is None:
                continue	# Hit hand no aligned residues with coordinates
            res, query_res = _alignment_residue_pairs(hit, hit_chain, query_chain)
            if len(res) >= 3:
                p, rms, npairs = alignment_transform(res, query_res, alignment_cutoff_distance)
                stats.append((rms, npairs))
                structure.position = p

    if log and query_chain is not None and align and stats:
        chain_id = hit.get('chain_id')
        cname = '' if chain_id is None else f' chain {chain_id}'
        if len(structures) == 1:
            msg = f'Alignment of {db_id}{cname} to query has RMSD {"%.3g" % rms} using {npairs} of {len(res)} paired residues'
        else:
            rms = [rms for rms,npair in stats]
            rms_min, rms_max = min(rms), max(rms)
            npair = [npair for rms,npair in stats]
            npair_min, npair_max = min(npair), max(npair)
            msg = f'Alignment of {db_id}{cname} ensemble of {len(stats)} structures to query has RMSD {"%.3g" % rms_min} - {"%.3g" % rms_max} using {npair_min}-{npair_max} of {len(res)} paired residues'      
        if alignment_cutoff_distance is not None and alignment_cutoff_distance > 0:
            msg += f' within cutoff distance {alignment_cutoff_distance}'
        session.logger.info(msg)

    structures = [s for s in structures if not s.deleted]
    return structures

def _remember_sequence_alignment(structure, query_chain, hit):
    '''
    Remember the positions of the aligned residues as part of the hit dictionary,
    so that after hit residues are trimmed we can still deduce the hit to query
    residue pairing.  The Foldseek hit and query alignment only includes a subset
    of residues from the structures and only include residues that have atomic coordinates.
    '''
    if hit.get('program') == 'foldseek':
        # Foldseek indexing only counts residues with atom coordinates
        chain_id = hit.get('chain_id')
        hit_chain = structure_chain_with_id(structure, chain_id)
        hit['aligned_residue_offsets'] = _indices_of_residues_with_coords(hit_chain, hit['tstart'], hit['tend'])
        hit['query_residue_offsets'] = _indices_of_residues_with_coords(query_chain, hit['qstart'], hit['qend'])
    else:
        hit['aligned_residue_offsets'] = list(range(hit['tstart']-1, hit['tend']))
        hit['query_residue_offsets'] = list(range(hit['qstart']-1, hit['qend']))

def _indices_of_residues_with_coords(chain, start, end):
    seqi = []
    si = 0
    for i, r in enumerate(chain.residues):
        if r is not None and r.name in _foldseek_accepted_3_letter_codes and r.find_atom('CA'):
            si += 1
            if si >= start and si <= end:
                seqi.append(i)
            elif si > end:
                break
    return seqi
    
def _trim_structure(structure, hit, trim, ligand_range = 3.0, log = True):
    if not trim:
        return

    chain_res, ri_start, ri_end = _residue_range(structure, hit, structure.session.logger)
    if ri_start is None:
        name = hit.get('database_full_id')
        msg = f'Hit {name} has no coordinates for aligned residues'
        structure.session.logger.warning(msg)

    msg = []
    logger = structure.session.logger  # Get logger before structure deleted.
    
    trim_lig = (trim is True or 'ligands' in trim) and hit['database'].startswith('pdb')
    chain_id = hit.get('chain_id')
    if (trim is True or 'chains' in trim) and chain_id is not None:
        if trim_lig:
            # Delete polymers in other chains, but not ligands from other chains.
            ocres = [c.existing_residues for c in tuple(structure.chains) if c.chain_id != chain_id]
            if ocres:
                from chimerax.atomic import concatenate
                ocres = concatenate(ocres)
        else:
            # Delete everything from other chain ids, including ligands.
            sres = structure.residues
            ocres = sres[sres.chain_ids != chain_id]
        if ocres:
            msg.append(f'{structure.num_chains-1} extra chains')
            ocres.delete()

    trim_seq = (trim is True or 'sequence' in trim)
    if trim_seq:
        if ri_start is None:
            chain_res.delete()
        else:
            # Trim end first because it changes the length of the res array.
            if ri_end < len(chain_res)-1:
                cterm = chain_res[ri_end+1:]
                msg.append(f'{len(cterm)} C-terminal residues')
                cterm.delete()
            if ri_start > 0:
                nterm = chain_res[:ri_start]
                msg.append(f'{len(nterm)} N-terminal residues')
                nterm.delete()

    if trim_lig:
        if not structure.deleted and structure.num_residues > len(chain_res):
            sres = structure.residues
            from chimerax.atomic import Residue
            npres = sres[sres.polymer_types == Residue.PT_NONE]
            if len(chain_res) == 0:
                npres.delete()
            else:
                npnear = _find_close_residues(npres, chain_res, ligand_range)
                npfar = npres.subtract(npnear)
                if npfar:
                    msg.append(f'{len(npfar)} non-polymer residues more than {ligand_range} Angstroms away')
                    npfar.delete()
                
    if log and msg:
        logger.info(f'Deleted {", ".join(msg)}.')


def _find_close_residues(residues1, residues2, distance):
    a1 = residues1.atoms
    axyz1 = a1.scene_coords
    axyz2 = residues2.atoms.scene_coords
    from chimerax.geometry import find_close_points
    ai1, ai2 = find_close_points(axyz1, axyz2, distance)
    r1near = a1[ai1].unique_residues
    return r1near

def _hit_chain(structure, hit):
    chain_id = hit.get('chain_id')
    if chain_id is None:
        chain = structure.chains[0] if structure.num_chains == 1 else None
    else:
        chain = structure_chain_with_id(structure, chain_id)
    return chain

def _residue_range(structure, hit, log):
    '''
    Return the residue range for the subset of residues used in the hit alignment.
    Foldseek tstart and tend target residue numbers are not PDB residue numbers.
    Instead they start at 1 and include only residues with atomic coordinates in the chain.
    '''
    chain_id = hit.get('chain_id')
    chain = structure_chain_with_id(structure, chain_id)
    if hit.get('program') == 'foldseek':
        # Foldseek index is for the sequence of residues with atomic coordinates
        ri0, ri1 = hit['tstart']-1, hit['tend']-1
        res = foldseek_alignment_residues(chain.existing_residues)
    else:
        # Non-foldseek index is for the full sequence including residues without coordinates.
        hro = hit['aligned_residue_offsets']
        ares = chain.residues[hro[0]:hro[-1]+1]
        eares = [r for r in ares if r is not None]
        res = chain.existing_residues
        if len(eares) == 0:
            # There are no hit residues with coordinates that are part of the alignment.
            ri0 = ri1 = None
        else:
            ri0 = res.index(eares[0])
            ri1 = res.index(eares[-1])
    return res, ri0, ri1

def _check_structure_sequence(structure, hit):
    # Check that the database structure sequence matches what Foldseek gives in the pairwise alignment.
    hri = hit['aligned_residue_offsets']
    chain = _hit_chain(structure, hit)
    struct_seq = chain.characters
    saligned_seq = ''.join(struct_seq[ri] for ri in hri)
    hseq_ungapped = hit['taln'].replace('-', '')
    # Don't warn when Foldseek has an X but ChimeraX has non-X for a modified residue.
    mismatches = [i for i, (saa, haa) in enumerate(zip(saligned_seq, hseq_ungapped)) if saa != haa and haa != 'X']
    if mismatches:
        i = mismatches[0]
        msg = f'Hit structure sequence {structure.name} amino acid {saligned_seq[i]} at sequence position {hri[i]} does not match sequence alignment {hseq_ungapped[i]} at position {i}.  Structure sequence {struct_seq}.  Alignment sequence {hseq_ungapped}.'
        structure.session.logger.warning(msg)

def structure_chain_with_id(structure, chain_id):
    if chain_id is None:
        chains = structure.chains
    else:
        chains = [chain for chain in structure.chains if chain.chain_id == chain_id]
    chain = chains[0] if len(chains) == 1 else None
    return chain

# Foldseek recognized residues from
#   https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/GemmiWrapper.cpp
_foldseek_accepted_3_letter_codes = set('ALA,ARG,ASN,ASP,CYS,GLN,GLU,GLY,HIS,ILE,LEU,LYS,MET,PHE,PRO,SER,THR,TRP,TYR,VAL,MSE,MLY,FME,HYP,TPO,CSO,SEP,M3L,HSK,SAC,PCA,DAL,CME,CSD,OCS,DPR,B3K,ALY,YCM,MLZ,4BF,KCX,B3E,B3D,HZP,CSX,BAL,HIC,DBZ,DCY,DVA,NLE,SMC,AGM,B3A,DAS,DLY,DSN,DTH,GL3,HY3,LLP,MGN,MHS,TRQ,B3Y,PHI,PTR,TYS,IAS,GPL,KYN,CSD,SEC,UNK'.split(','))

def foldseek_alignment_residues(residues):
    '''
    Foldseek omits some non-standard residues.  It appears the ones it accepts are about 150
    hardcoded in a C++ file

        ​https://github.com/steineggerlab/foldseek/blob/master/lib/gemmi/resinfo.hpp
    '''
    rok = [r for r in residues if r.name in _foldseek_accepted_3_letter_codes]
    if len(rok) < len(residues):
        from chimerax.atomic import Residues
        residues = Residues(rok)
    if (residues.atoms.names == 'CA').sum() < len(residues):
        from chimerax.atomic import Residues
        residues = Residues([r for r in residues if 'CA' in r.atoms.names])
    return residues
    
def _alignment_residue_pairs(hit, hit_chain, query_chain):
    hres = hit_chain.residues
    qres = query_chain.residues
    hro = hit['aligned_residue_offsets']
    qro = hit['query_residue_offsets']
    
    qaln, taln = hit['qaln'], hit['taln']
    ti = qi = 0
    atr, aqr = [], []
    for qaa, taa in zip(qaln, taln):
        if qaa != '-' and taa != '-':
            qr = qres[qro[qi]]
            tr = hres[hro[ti]]
            if tr is not None and qr is not None:
                atr.append(tr)
                aqr.append(qr)
        if taa != '-':
            ti += 1
        if qaa != '-':
            qi += 1

    from chimerax.atomic import Residues
    return Residues(atr), Residues(aqr)

def sequence_alignment(hits, qstart, qend):
    '''
    Return the sequence alignment between hits and query as a numpy 2D array of bytes.
    The first row is the query sequence with one subsequent row for each hit.
    There are no gaps in the query, and gaps in the hit sequences have 0 byte values.
    '''
    nhit = len(hits)
    qlen = qend - qstart + 1
    from numpy import zeros, byte
    alignment = zeros((nhit+1, qlen), byte)
    for h, hit in enumerate(hits):
        qaln, taln = hit['qaln'], hit['taln']
        qi = hit['qstart']
        for qaa, taa in zip(qaln, taln):
            if qi >= qstart and qi <= qend:
                if qaa != '-':
                    ai = qi-qstart
                    alignment[0,ai] = ord(qaa)	# First row is query sequence
                    if taa != '-':
                        alignment[h+1,ai] = ord(taa)
            if qaa != '-':
                qi += 1
    return alignment

def alignment_coordinates(hits, qstart, qend):
    '''
    Return C-alpha atom coordinates for aligned sequences.
    Also return a mask indicating positions that are not sequence gaps.
    '''
    # TODO: This code only handles Foldseek hits where sequences do not include
    #  residues that lack coordinates.  For mmseqs2 or blast results the code
    #  needs to be fixed to account for the sequences including residues without coordinates.
    nhit = len(hits)
    qlen = qend - qstart + 1
    from numpy import zeros, float32
    xyz = zeros((nhit, qlen, 3), float32)
    mask = zeros((nhit, qlen), bool)
    for h, hit in enumerate(hits):
        qaln, taln = hit['qaln'], hit['taln']
        qi = hit['qstart']
        hi = hit['tstart']
        hxyz = hit_coords(hit)
        for qaa, taa in zip(qaln, taln):
            if qaa != '-' and taa != '-' and qi >= qstart and qi <= qend:
                ai = qi-qstart
                xyz[h,ai,:] = hxyz[hi-1]
                mask[h,ai] = True
            if qaa != '-':
                qi += 1
            if taa != '-':
                hi += 1
    return xyz, mask

def alignment_transform(res, query_res, cutoff_distance = None):
    qatoms = query_res.find_existing_atoms('CA')
    qxyz = qatoms.scene_coords
    tatoms = res.find_existing_atoms('CA')
    txyz = tatoms.coords
    p, rms, npairs = align_xyz_transform(txyz, qxyz, cutoff_distance = cutoff_distance)
    return p, rms, npairs

def align_xyz_transform(txyz, qxyz, cutoff_distance = None):
    if cutoff_distance is None or cutoff_distance <= 0:
        from chimerax.geometry import align_points
        p, rms = align_points(txyz, qxyz)
        npairs = len(txyz)
    else:
        from chimerax.std_commands.align import align_and_prune
        p, rms, indices = align_and_prune(txyz, qxyz, cutoff_distance)
#        look_for_more_alignments(txyz, qxyz, cutoff_distance, indices)
        npairs = len(indices)
    return p, rms, npairs

def look_for_more_alignments(txyz, qxyz, cutoff_distance, indices):
    sizes = [len(txyz), len(indices)]
    from numpy import ones
    mask = ones(len(txyz), bool)
#    from chimerax.std_commands.align import align_and_prune, IterationError
    while True:
        mask[indices] = 0
        if mask.sum() < 3:
            break
        try:
#            p, rms, indices = align_and_prune(txyz, qxyz, cutoff_distance, mask.nonzero()[0])
            p, rms, indices, close_mask = align_and_prune(txyz, qxyz, cutoff_distance, mask.nonzero()[0])
            sizes.append(len(indices))
            nclose = close_mask.sum()
            if nclose > len(indices):
                sizes.append(-nclose)
                sizes.append(-(mask*close_mask).sum())
        except IterationError:
            break
    print (' '.join(str(s) for s in sizes))

from chimerax.core.errors import UserError
class IterationError(UserError):
    pass

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
        dxyz = p*xyz - ref_xyz
        d2 = (dxyz*dxyz).sum(axis=1)
        close_mask = (d2 <= cutoff2)
        return p, rms, indices, close_mask

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

def hit_coords(hit):
    '''
    Returns the C-alpha atom positions for each residue that has such coordinates in the hit structure
    including residues outsides the aligned interval.
    '''
    hxyz = hit.get('tca')
    return hxyz

def hit_residue_pairing(hit):
    if hit.get('program') == 'foldseek':
        return hit_residue_pairing_foldseek(hit)
    return hit_residue_pairing_blast(hit)

def hit_residue_pairing_foldseek(hit):
    '''
    Returns an array of hit indices and query indices that are paired that index into the
    C-alpha coordinates array for the hit and the query.  Those coordinate arrays include
    all residues which have C-alpha coordinates even outside the alignment interval for the hit.
    '''
    ati, aqi = _alignment_index_pairs(hit, offset = True)
    return ati, aqi

def hit_residue_pairing_blast(hit):
    '''
    Returns an array of hit indices and query indices that are paired that index into the
    C-alpha coordinates array for the hit and the query.  Those coordinate arrays include
    all residues which have C-alpha coordinates even outside the alignment interval for the hit.
    BLAST can align residues without coordinates and uses full sequence indexing, unlike
    foldseek which only considers residues with coordinates and uses indexing only relative to those residues.
    '''
    ati, aqi = _alignment_index_pairs(hit, offset = True)  # Indices relative to full sequences

    # Convert to indices relative to coordinate arrays
    # Paired residues may not have coordinates and those are eliminated.
    c2f = hit.get('tca_index')  # Array mapping coordinate index to full sequence index
    if c2f is None:
        return None, None
    f2c = {fi:ci for ci, fi in enumerate(c2f)}
    qc2f = hit['qcoord_index']  # Array mapping coordinate index to full sequence index for query
    qf2c = {fi:ci for ci, fi in enumerate(qc2f)}
    cti, cqi = [], []
    for ti,qi in zip(ati, aqi):
        if ti in f2c and qi in qf2c:
            cti.append(f2c[ti])
            cqi.append(qf2c[qi])
    return cti, cqi

def _alignment_index_pairs(hit, offset):
    '''
    Return an array of hit indices and query indices that are paired that index into
    the hit and query ungapped sequences starting at the start of the alignment.
    If offset is true then indexing is shifted by hit['tstart'] and hit['qstart']
    to give full sequence indexing (or for foldseek residue with coordinates indexing).
    '''
    qaln, taln = hit['qaln'], hit['taln']
    ti = qi = 0
    ati, aqi = [], []
    for qaa, taa in zip(qaln, taln):
        if qaa != '-' and taa != '-':
            ati.append(ti)
            aqi.append(qi)
        if taa != '-':
            ti += 1
        if qaa != '-':
            qi += 1
    from numpy import array, int32
    ati, aqi = array(ati, int32), array(aqi, int32)
    if offset:
        ati += hit['tstart']-1
        aqi += hit['qstart']-1
    return ati, aqi

def _show_ribbons(structure):
    for c in structure.chains:
        cres = c.existing_residues
        if not cres.ribbon_displays.any():
            cres.ribbon_displays = True
            cres.atoms.displays = False

def open_foldseek_m8(session, path, query_chain = None, database = None):
    with open(path, 'r') as file:
        hit_lines = file.readlines()

    if query_chain is None:
        query_chain, models = _guess_query_chain(session, path)

    if database is None:
        database = _guess_database(path, hit_lines)
        if database is None:
            from os.path import basename
            filename = basename(path)
            from chimerax.core.errors import UserError
            raise UserError('Cannot determine database for foldseek file {filename}.  Specify which database ({", ".join(foldseek_databases)}) using the open command "database" option, for example "open {filename} database pdb100".')

    # TODO: The query_chain model has not been added to the session yet.  So it is has no model id which
    # messes up the hits table header.  Also it causes an error trying to set the Chain menu if that menu
    # has already been used.  But I don't want to add the structure to the session because then its log output
    # appears in the open Notes table.
    if models:
        session.models.add(models)
        models = []

    from .search import parse_search_result
    hits = [parse_search_result(hit, database) for hit in hit_lines]
    results = FoldseekResults(hits, database, query_chain)
    from .gui import show_foldseek_results
    show_foldseek_results(session, results)

    return models, ''

def _guess_query_chain(session, results_path):
    '''Look for files in same directory as foldseek results to figure out query chain.'''

    from os.path import dirname, join, exists
    results_dir = dirname(results_path)
    qpath = join(results_dir, 'query')
    if exists(qpath):
        # See if the original query structure file is already opened.
        path, chain_id = open(qpath,'r').read().split('\t')
        from chimerax.atomic import AtomicStructure
        for s in session.models.list(type = AtomicStructure):
            if hasattr(s, 'filename') and s.filename == path:
                for c in s.chains:
                    if c.chain_id == chain_id:
                        return c, []
        # Try opening the original query structure file
        if exists(path):
            models, status = session.open_command.open_data(path)
            chains = [c for c in models[0].chains if c.chain_id == chain_id]
            return chains[0], models

    qpath = join(results_dir, 'query.cif')
    if exists(qpath):
        # Use the single-chain mmCIF file used in the Foldseek submission
        models, status = session.open_command.open_data(qpath)
        return models[0].chains[0], models
    
    return None, []

def _guess_database(path, hit_lines):
    from .search import foldseek_databases
    for database in foldseek_databases:
        if path.endswith(database + '.m8'):
            return database

    from os.path import dirname, join, exists
    dpath = join(dirname(path), 'database')
    if exists(dpath):
        database = open(dpath,'r').read()
        return database

    hit_file = hit_lines[0].split('\t')[1]
    if '.cif.gz' in hit_file:
        database = 'pdb100'

    return None

class FoldseekResults:
    def __init__(self, hits, database, query_chain, trim = True, alignment_cutoff_distance = 2.0,
                 program = 'foldseek'):
        self.hits = hits
        self.database = database
        self._query_chain = query_chain
        self._program = program		# Hit indices interpretation depends on program
        
        # Default values used when opening and aligning structures
        self.trim = trim
        self.alignment_cutoff_distance = alignment_cutoff_distance

        # Cached values
        self._clear_cached_values()

        r2i = {r:i for i,r in enumerate(query_chain.residues) if r is not None}
        qc2f = [r2i[r] for r in self.query_residues]  # Query coordinate index to full sequence index
        for hit in hits:
            hit['qcoord_index'] = qc2f
            hit['program'] = program

    def _clear_cached_values(self):
        self._query_residues = None
        self._query_alignment_range = None
        self._sequence_alignment_array = None
        self._alignment_coordinates = None
        self._lddt_score_array = None

    def replace_hits(self, hits):
        self.hits = hits
        self._clear_cached_values()

    @property
    def num_hits(self):
        return len(self.hits)

    @property
    def session(self):
        qc = self.query_chain
        return qc.structure.session if qc else None

    @property
    def query_chain(self):
        qc = self._query_chain
        if qc is not None and qc.structure is None:
            self._query_chain = qc = None
        return qc

    @property
    def query_residues(self):
        qres = self._query_residues
        if qres is None:
            if self._program == 'foldseek':
                qres = foldseek_alignment_residues(self._query_chain.existing_residues)
            else:
                qres = self._query_chain.existing_residues
            self._query_residues = qres
        return qres

    def query_alignment_range(self):
        '''Return the range of query residue numbers (qstart, qend) that includes all the hit alignments.'''
        # TODO: Some uses of this assume the range uses sequence numbering, and other uses like lddt
        #  assume it uses the numbering of residues with coordinates.  Those are the same for Foldseek
        #  but not for mmseqs2 or blast.
        qar = self._query_alignment_range
        if qar is not None:
            return qar
        qstarts = []
        qends = []
        for hit in self.hits:
            qstarts.append(hit['qstart'])
            qends.append(hit['qend'])
        qar = qstart, qend = min(qstarts), max(qends)
        return qar

    def have_c_alpha_coordinates(self):
        for hit in self.hits:
            if 'tca' not in hit:
                return False
        return True

    def open_hit(self, session, hit):
        open_hit(self.session, hit, self.query_chain, trim = self.trim,
                 alignment_cutoff_distance = self.alignment_cutoff_distance)

    def sequence_alignment_array(self):
        saa = self._sequence_alignment_array
        if saa is not None:
            return saa
        qstart, qend = self.query_alignment_range()
        self._sequence_alignment_array = saa = sequence_alignment(self.hits, qstart, qend)
        return saa

    def alignment_coordinates(self):
        ac = self._alignment_coordinates
        if ac is not None:
            return ac
        qstart, qend = self.query_alignment_range()
        self._alignment_coordinates = hits_xyz, hits_mask = alignment_coordinates(self.hits, qstart, qend)
        return hits_xyz, hits_mask
    
    def compute_rmsds(self, alignment_cutoff_distance = None):
        if self.query_chain is None:
            return False
        # Compute percent coverage and percent close C-alpha values per hit.
        qres = self.query_residues
        qatoms = qres.find_existing_atoms('CA')
        query_xyz = qatoms.coords
        for hit in self.hits:
            hi, qi = hit_residue_pairing(hit)
            if hi is not None and qi is not None:
                hxyz = hit_coords(hit)
                if hxyz is not None:
                    p, rms, npairs = align_xyz_transform(hxyz[hi], query_xyz[qi],
                                                         cutoff_distance=alignment_cutoff_distance)
                    hit['rmsd'] = rms
                    hit['close'] = 100*npairs/len(hi)
                    hit['cutoff_distance'] = alignment_cutoff_distance
                hit['coverage'] = 100 * len(qi) / len(query_xyz)
        return True

    def set_coverage_attribute(self):
        if self.query_chain is None  or getattr(self, '_coverage_attribute_set', False):
            return
        self._coverage_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'foldseek_coverage', "Foldseek", attr_type = int)

        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            if ri >= qstart-1 and ri <= qend-1:
                ai = ri-(qstart-1)
                alignment_array = self.sequence_alignment_array()
                count = count_nonzero(alignment_array[1:,ai])
            else:
                count = 0
            r.foldseek_coverage = count

    def set_conservation_attribute(self):
        if self.query_chain is None  or getattr(self, '_conservation_attribute_set', False):
            return
        self._conservation_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'foldseek_conservation', "Foldseek", attr_type = float)

        alignment_array = self.sequence_alignment_array()
        seq, count, total = _consensus_sequence(alignment_array)
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            if ri >= qstart-1 and ri <= qend-1:
                ai = ri-(qstart-1)
                conservation = count[ai] / total[ai]
            else:
                conservation = 0
            r.foldseek_conservation = conservation

    def set_entropy_attribute(self):
        if self.query_chain is None or getattr(self, '_entropy_attribute_set', False):
            return
        self._entropy_attribute_set = True
        
        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'foldseek_entropy', "Foldseek", attr_type = float)

        alignment_array = self.sequence_alignment_array()
        entropy = _sequence_entropy(alignment_array)
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            if ri >= qstart-1 and ri <= qend-1:
                ai = ri-(qstart-1)
                r.foldseek_entropy = entropy[ai]

    def set_lddt_attribute(self):
        if self.query_chain is None or getattr(self, '_lddt_attribute_set', False):
            return
        self._lddt_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'foldseek_lddt', "Foldseek", attr_type = float)

        lddt_scores = self.lddt_scores()
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            if ri >= qstart-1 and ri <= qend-1:
                ai = ri-(qstart-1)
                alignment_array = self.sequence_alignment_array()
                nscores = count_nonzero(alignment_array[1:,ai])
                ave_lddt = lddt_scores[:,ai].sum() / nscores
            else:
                ave_lddt = 0
            r.foldseek_lddt = ave_lddt

    def lddt_scores(self):
        lddt_scores = self._lddt_score_array
        if lddt_scores is None:
            qstart, qend = self.query_alignment_range()
            qres = self.query_residues
            qatoms = qres.find_existing_atoms('CA')
            query_xyz = qatoms.coords[qstart-1:qend,:]
            hits_xyz, hits_mask = self.alignment_coordinates()
            from . import lddt
            lddt_scores = lddt.local_distance_difference_test(query_xyz, hits_xyz, hits_mask)
            self._lddt_score_array = lddt_scores
        return lddt_scores

    def take_snapshot(self, session, flags):
        data = {'hits': self.hits,
                'query_chain': self.query_chain,
                'database': self.database,
                'trim': self.trim,
                'alignment_cutoff_distance': self.alignment_cutoff_distance,
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        return FoldseekResults(data['hits'], data['database'], data['query_chain'],
                               trim = data.get('trim', True),
                               alignment_cutoff_distance = data.get('alignment_cutoff_distance', 2.0))


# -----------------------------------------------------------------------------
#
def foldseek_results(session):
    '''Return currently open foldseek results.'''
    from .gui import foldseek_panel
    fp = foldseek_panel(session)
    if fp:
        return fp.results

    from .blast import blast_results
    br = blast_results(session)
    if br:
        return br
    
    return None

# -----------------------------------------------------------------------------
#
def _consensus_sequence(alignment_array):
    seqlen = alignment_array.shape[1]
    from numpy import count_nonzero, bincount, argmax, empty, byte, int32
    seq = empty((seqlen,), byte)
    count = empty((seqlen,), int32)
    total = empty((seqlen,), int32)
    for i in range(seqlen):
        aa = alignment_array[:,i]
        total[i] = t = count_nonzero(aa)
        if t > 0:
            bc = bincount(aa)
            mi = argmax(bc[1:]) + 1
            seq[i] = mi
            count[i] = bc[mi]
        else:
            # No hit aligns with this column.
            seq[i] = count[i] = 0
    return seq, count, total

# -----------------------------------------------------------------------------
#
def _sequence_entropy(alignment_array):
    seqlen = alignment_array.shape[1]
    from numpy import bincount, empty, float32, array, int32
    entropy = empty((seqlen,), float32)
    aa_1_letter_codes = 'ARNDCEQGHILKMFPSTWYV'
    aa_char = array([ord(c) for c in aa_1_letter_codes], int32)
    bins = aa_char.max() + 1
    for i in range(seqlen):
        aa = alignment_array[:,i]
        bc = bincount(aa, minlength = bins)[aa_char]
        entropy[i] = _entropy(bc)
    return entropy
    
# -----------------------------------------------------------------------------
#
def _entropy(bin_counts):
    total = bin_counts.sum()
    if total == 0:
        return 0.0
    nonzero = (bin_counts > 0)
    p = bin_counts[nonzero] / total
    from numpy import log2
    e = -(p*log2(p)).sum()
    return e
    
# -----------------------------------------------------------------------------
#
def foldseek_open(session, hit_name, trim = None, align = True, alignment_cutoff_distance = None,
                  in_file_history = True, log = True):
    results = foldseek_results(session)
    if results is None:
        from chimerax.core.errors import UserError
        raise UserError('No foldseek results are available')
    query_chain = results.query_chain
    if trim is None:
        trim = results.trim
    if alignment_cutoff_distance is None:
        alignment_cutoff_distance = results.alignment_cutoff_distance
    matches = [hit for hit in results.hits if hit['database_full_id'] == hit_name]
    for hit in matches:
        open_hit(session, hit, query_chain, trim = trim,
                 align = align, alignment_cutoff_distance = alignment_cutoff_distance,
                 in_file_history = in_file_history, log = log)
    if len(matches) == 0:
        session.logger.error(f'No foldseek hit with name {hit_name}')

def foldseek_hit_by_name(session, hit_name):
    results = foldseek_results(session)
    if results is None:
        from chimerax.core.errors import UserError
        raise UserError('No foldseek results table is open')
    for hit in results.hits:
        if hit['database_full_id'] == hit_name:
            return hit, results
    return None, None
    
def foldseek_pairing(session, hit_structure, color = None, radius = None,
                     halfbond_coloring = None):
    '''Show pseudobonds between foldseek hit and query paired residues.'''
    hit_name = hit_structure.name
    hit, results = foldseek_hit_by_name(session, hit_name)
    if hit is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find any Foldseek hit {hit_name}')

    query_chain = results.query_chain
    r_pairs = hit_and_query_residue_pairs(hit_structure, query_chain, hit)
    ca_pairs = []
    for hr, qr in r_pairs:
        hca = hr.find_atom('CA')
        qca = qr.find_atom('CA')
        if hca and qca:
            ca_pairs.append((hca, qca))

    if len(ca_pairs) == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find any residues to pair for Foldseek hit {hit_name}')

    g = session.pb_manager.get_group(f'{hit_name} pairing')
    if g.id is not None:
        g.clear()

    for hca, qca in ca_pairs:
        b = g.new_pseudobond(hca, qca)
        if color is not None:
            b.color = color
        if radius is not None:
            b.radius = radius
        if halfbond_coloring is not None:
            b.halfbond = halfbond_coloring

    if g.id is None:
        session.models.add([g])

    return g

def hit_and_query_residue_pairs(hit_structure, query_chain, hit):
    ho = hit.get('aligned_residue_offsets')
    qo = hit.get('query_residue_offsets')
    if ho is None or qo is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Could not find Foldseek alignment offsets for {hit["database_id"]}')
    
    qres = query_chain.residues
    hit_chain = structure_chain_with_id(hit_structure, hit.get('chain_id'))
    hres = hit_chain.residues
    ahi, aqi = _alignment_index_pairs(hit, offset = False)
    r_pairs = []
    for hi, qi in zip(ahi, aqi):
        hr = hres[ho[hi]]
        qr = qres[qo[qi]]
        if hr is not None and qr is not None:
            r_pairs.append((hr, qr))

    return r_pairs
    
def foldseek_sequence_alignment(session, hit_structure):
    '''Show pairwise sequence alignment returned by Foldseek.'''
    hit_name = hit_structure.name
    hit, results = foldseek_hit_by_name(session, hit_name)
    if hit is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find any Foldseek hit {hit_name}')

    # Create query and hit gapped aligned sequences
    query_chain = results.query_chain
    qname = f'{query_chain.structure.name}_{query_chain.chain_id}'
    from chimerax.atomic import Sequence, SeqMatchMap
    qseq = Sequence(name = qname, characters = hit['qaln'])
    hseq = Sequence(name = hit_name, characters = hit['taln'])

    matches = len([1 for qc,tc in zip(hit['qaln'], hit['taln']) if qc == tc])
    paired = len([1 for qc,tc in zip(hit['qaln'], hit['taln']) if qc != '-' and tc != '-'])
    qlen = len([1 for qc in hit['qaln'] if qc != '-'])
    tlen = len([1 for qc in hit['taln'] if qc != '-'])
#    print (f'{matches} identical, {len(hit["qaln"])} gapped alignment length, query length {qlen}, hit length {tlen}, paired {paired}')

    # Create alignment
    seqs = [qseq, hseq]
    am = session.alignments
    a = am.new_alignment(seqs, identify_as = hit_name, name = f'Foldseek query {qname} and hit {hit_name}',
                         auto_associate = False, intrinsic = True)

    # Create query structure association with sequence
    reassoc = None  # Not used
    errors = 0
    query_match_map = SeqMatchMap(qseq, query_chain)
    qres = query_chain.residues
    for pos,qo in enumerate(hit.get('query_residue_offsets')):
        if qres[qo] is not None:
            query_match_map.match(qres[qo], pos)
    a.prematched_assoc_structure(query_match_map, errors, reassoc)  # Associate query

    # Create hit structure association with sequence
    hit_chain = structure_chain_with_id(hit_structure, hit.get('chain_id'))
    hit_match_map = SeqMatchMap(hseq, hit_chain)
    hres = hit_chain.residues
    for pos,ho in enumerate(hit.get('aligned_residue_offsets')):
        if hres[ho] is not None:
            hit_match_map.match(hres[ho], pos)
    a.prematched_assoc_structure(hit_match_map, errors, reassoc)  # Associate hit

    return a
        
def register_foldseek_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, Or, ListOf, FloatArg, Color8Arg, StringArg
    from chimerax.atomic import ChainArg, StructureArg
    TrimArg = Or(ListOf(EnumOf(['chains', 'sequence', 'ligands'])), BoolArg)

    desc = CmdDesc(
        required = [('hit_name', StringArg)],
        keyword = [('trim', TrimArg),
                   ('align', BoolArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('in_file_history', BoolArg),
                   ('log', BoolArg),
                   ],
        synopsis = 'Open Foldseek result structure and align to query'
    )
    register('foldseek open', desc, foldseek_open, logger=logger)

    desc = CmdDesc(
        required = [('hit_structure', StructureArg)],
        keyword = [('color', Color8Arg),
                   ('radius', FloatArg),
                   ('halfbond_coloring', BoolArg),
                   ],
        synopsis = 'Show Foldseek result table row'
    )
    register('foldseek pairing', desc, foldseek_pairing, logger=logger)

    desc = CmdDesc(
        required = [('hit_structure', StructureArg)],
        synopsis = 'Show Foldseek sequence alignment for one hit'
    )
    register('foldseek seqalign', desc, foldseek_sequence_alignment, logger=logger)
