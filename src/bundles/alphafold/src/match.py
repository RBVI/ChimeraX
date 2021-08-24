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

def alphafold_match(session, sequences, color_confidence=True, trim = True,
                    search = True, ignore_cache=False):
    '''
    Find the most similar sequence in the AlphaFold database to each specified
    sequence and load them.  The specified sequences can be Sequence instances
    or Chain instances.  For chains the loaded structure will be aligned to
    the chain structure.
    '''
    from chimerax.atomic import Residue
    sequences = [seq for seq in sequences
                 if not hasattr(seq, 'polymer_type') or seq.polymer_type == Residue.PT_AMINO]
    if len(sequences) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No protein sequences specified')

    log = session.logger

    # Use UniProt identifiers in file metadata to get AlphaFold models.
    seq_models = _fetch_by_uniprot_id(session, sequences, color_confidence=color_confidence,
                                      trim=trim, local=(search == 'local'), log=log,
                                      ignore_cache=ignore_cache)
        
    # Try sequence search if some sequences were not found by UniProt identifier.
    if search:
        search_seqs = [seq for seq in sequences if seq not in seq_models]
        search_seq_models = _fetch_by_sequence(session, search_seqs,
                                               color_confidence=color_confidence,
                                               trim=trim, local=(search == 'local'),
                                               log=log, ignore_cache=ignore_cache)
        seq_models.update(search_seq_models)

    # Report sequences with no AlphaFold model
    seqs_no_model = [seq for seq in sequences if seq not in seq_models]
    if seqs_no_model:
        msg = ('No AlphaFold model with similar sequence for %s'
               % _sequences_description(seqs_no_model))
        log.warning(msg)

    mlist, nmodels = _group_chains_by_structure(seq_models)
    session.models.add(mlist)

    msg = 'Opened %d AlphaFold model%s' % (nmodels, _plural(nmodels))
    log.info(msg)

    return mlist

def _fetch_by_uniprot_id(session, sequences, color_confidence = True, trim = True,
                         local = False, ignore_cache = False, log = None):
    # Get Uniprot ids from mmCIF or PDB file metadata
    sequence_uids = _sequence_uniprot_ids(sequences)
    seq_models, missing_uids = _alphafold_models(session, sequences, sequence_uids,
                                                 color_confidence=color_confidence, trim=trim,
                                                 ignore_cache=ignore_cache)
    if missing_uids:
        log.warning('%d UniProt id%s'
                    % (len(missing_uids), _plural(missing_uids)) +
                    (' do not have AlphaFold database models: ' if len(missing_uids) > 1 else
                     ' does not have an AlphaFold database model: ') +
                    _uniprot_sequences_description(missing_uids))
    if seq_models:
        uid_seqs = _uniprot_sequences(seq_models)
        log.info('%d AlphaFold model%s found using UniProt identifier%s: %s'
                 % (len(uid_seqs), _plural(uid_seqs), _plural(uid_seqs),
                    _uniprot_sequences_description(uid_seqs)))
    return seq_models

def _fetch_by_sequence(session, sequences, color_confidence = True, trim = True,
                       local = False, ignore_cache = False, log = None):
    
    from .search import alphafold_sequence_search, SearchError
    seq_strings = [seq.characters for seq in sequences]
    try:
        seq_uids = alphafold_sequence_search(seq_strings, local = local, log = log)
    except SearchError as e:
        log.error(str(e))
        return {}

    seq_uids = [(seq,uid) for seq, uid in zip(sequences, seq_uids) if uid is not None]
    from chimerax.atomic import Chain
    for seq, uid in seq_uids:
        if isinstance(seq, Chain):
            uid.chain_id = seq.chain_id
    
    seq_models, missing_uids = \
        _alphafold_models(session, sequences, seq_uids,
                          color_confidence=color_confidence, trim=trim,
                          ignore_cache=ignore_cache)
    if missing_uids and log:
        missing_names = ', '.join('%s (%s)' % (uid, _sequences_description(seqs))
                                  for uid,seqs in missing_uids.items())
        log.warning('Sequence search found %d UniProt id%s'
                    % (len(missing_uids), _plural(missing_uids)) +
                    ' that do not have AlphaFold database models: %s' % missing_names)
    if seq_models and log:
        uid_seqs = _uniprot_sequences(seq_models)
        log.info('%d AlphaFold model%s found using sequence similarity searches: %s'
                 % (len(uid_seqs), _plural(uid_seqs), _uniprot_sequences_description(uid_seqs)))

    return seq_models

def _alphafold_models(session, sequences, seq_uids, color_confidence=True, trim=True,
                      ignore_cache=False):
    seq_models = {}
    missing = {}
    from chimerax.core.errors import UserError
    from .fetch import alphafold_fetch
    from chimerax.atomic import Chain
    for seq, uid in seq_uids:
        if uid.uniprot_id in missing:
            missing[uid.uniprot_id].append(seq)
            continue
        try:
            models, status = alphafold_fetch(session, uid.uniprot_id,
                                             color_confidence=color_confidence,
                                             add_to_session=False,
                                             ignore_cache=ignore_cache)
        except UserError as e:
            if not str(e).endswith('Not Found'):
                session.logger.warning(str(e))
            missing[uid.uniprot_id] = [seq]
            models = []
        for alphafold_model in models:
            alphafold_model._log_info = False          # Don't log chain tables
            alphafold_model.uniprot_id = uid.uniprot_id
            alphafold_model.uniprot_name = uid.uniprot_name
            seq_match = getattr(uid, 'range_from_sequence_match', False)
            if trim and not seq_match and uid.database_sequence_range:
                _trim_sequence(alphafold_model, uid.database_sequence_range)
            if isinstance(seq, Chain):
                alphafold_model.observed_num_res = seq.num_existing_residues
                _rename_chains(alphafold_model, seq)
                _align_to_chain(alphafold_model, seq)
            if trim and seq_match:
                seq_range = getattr(alphafold_model, 'seq_match_range', None)
                if seq_range:
                    _trim_sequence(alphafold_model, seq_range)
            alphafold_model.name = 'UniProt %s' % uid.uniprot_id
            if isinstance(seq, Chain):
                alphafold_model.name += ' chain %s' % seq.chain_id
        if models:
            if seq in seq_models:
                seq_models[seq].extend(models)
            else:
                seq_models[seq] = models

    return seq_models, missing

def _trim_sequence(structure, sequence_range):
    seq_start, seq_end = sequence_range
    rdelete = []
    for chain in structure.chains:
        res = chain.existing_residues
        rnums = res.numbers
        from numpy import logical_or
        rmask = logical_or(rnums < seq_start, rnums > seq_end)
        if rmask.any():
            rdelete.append(res[rmask])
    if rdelete:
        from chimerax.atomic import concatenate
        rdel = concatenate(rdelete)
        rdel.delete()
        
def _rename_chains(structure, chain):
    schains = structure.chains
    if len(schains) > 1:
        cnames = ', '.join(c.chain_id for c in schains)
        structure.session.logger.warning('Alphafold structure %s has %d chains (%s), expected 1.  Not renaming chain id to match target structure.' % (structure.name, len(schains), cnames))

    for schain in schains:
        schain.chain_id = chain.chain_id
        
def _align_to_chain(structure, chain):
    from chimerax.match_maker.match import cmd_match
    results = cmd_match(structure.session, structure.atoms, to = chain.existing_residues.atoms,
                        verbose=None)
    if len(results) == 1:
        r = results[0]
        rmsd = r.get('full RMSD')
        if rmsd is not None:
            structure.rmsd = rmsd
        rseq, mseq = r.get('aligned ref seq'), r.get('aligned match seq')
        if rseq and mseq:
            range = _sequence_match_range(rseq, mseq)
            if range:
                structure.seq_match_range = range
            structure.seq_identity = _sequence_identity(rseq, mseq, range)

def _sequence_identity(seq1, seq2, range2 = None):
    if range2:
        rmin, rmax = range2
        pairs = [(aa1,aa2) for aa1, aa2, r2 in zip(seq1.characters, seq2.characters, seq2.residues)
                 if r2 is not None and r2.number >= rmin and r2.number <= rmax]
    else:
        pairs = list(zip(seq1.characters, seq2.characters))

    m = 0
    for aa1, aa2 in pairs:
        if aa1 == aa2 and aa1 != '.':
            m += 1

    len1 = len2 = 0
    for aa1, aa2 in pairs:
        if aa1 != '.':
            len1 += 1
        if aa2 != '.':
            len2 += 1
        
    d = min(len1, len2)
    return m/d if d > 0 else 0.0

def _sequence_match_range(seq1, seq2):
    '''
    Return the residue number range for seq2 spanning
    the first to last matching position in the sequence alignment.
    '''
    rnum1 = rnum2 = None
    for aa1, aa2, r2 in zip(seq1.characters, seq2.characters, seq2.residues):
        if aa1 != '.' and aa2 != '.' and r2 is not None:
            rnum2 = r2.number
            if rnum1 is None:
                rnum1 = rnum2
    if rnum1 is None:
        return None
    return (rnum1, rnum2)

def _sequence_uniprot_ids(sequences):
    seq_uids = []
    from chimerax.atomic import uniprot_ids
    for structure, schains in _chains_by_structure(sequences).items():
        id_to_chain = {chain.chain_id:chain for chain in schains}
        for u in uniprot_ids(structure):
            chain = id_to_chain.get(u.chain_id)
            if chain:
                seq_uids.append((chain, u))
    for seq in sequences:
        if hasattr(seq, 'uniprot_accession'):
            from .search import UniprotSequence
            u = UniprotSequence(seq.uniprot_accession, getattr(seq, 'uniprot_name', None),
                                None, None)
            seq_uids.append((seq, u))
    return seq_uids

def _chains_by_structure(sequences):
    struct_chains = {}
    from chimerax.atomic import Chain
    for chain in sequences:
        if isinstance(chain, Chain):
            s = chain.structure
            if s in struct_chains:
                struct_chains[s].append(chain)
            else:
                struct_chains[s] = [chain]
    return struct_chains

def _uniprot_sequences_description(uniprot_sequences):
    info = ', '.join('%s (%s)' % (uid, _sequences_description(seqs))
                     for uid,seqs in uniprot_sequences.items())
    return info

def _uniprot_sequences(seq_models):
    uc = {}
    for seq, models in seq_models.items():
        for model in models:
            uid = model.uniprot_id
            if uid in uc:
                uc[uid].append(seq)
            else:
                uc[uid] = [seq]
    return uc

def _sequences_description(seqs):
    from chimerax.atomic import Chain
    cids, uids, nseq = [], [], 0
    for seq in seqs:
        if isinstance(seq, Chain):
            cids.append(seq.chain_id)
        elif hasattr(seq, 'uniprot_name'):
            uids.append(seq.uniprot_name)
        elif hasattr(seq, 'uniprot_accession'):
            uids.append(seq.uniprot_accession)
        else:
            nseq += 1
    items = []
    if cids:
        items.append('chain%s %s' % (_plural(cids), ','.join(cids)))
    if uids:
        items.append('UniProt %s' % ', '.join(uids))
    if nseq > 0:
        items.append('%d sequences' % nseq)
    descrip = ', '.join(items)
    return descrip

def _plural(seq):
    n = seq if isinstance(seq, int) else len(seq)
    return 's' if n > 1 else ''

def _group_chains_by_structure(seq_models):
    
    # Group models by structure
    struct_models = {}
    seq_only_models = []
    from chimerax.atomic import Chain
    for chain,models in seq_models.items():
        if isinstance(chain, Chain):
            s = chain.structure
            if s in struct_models:
                struct_models[s].extend(models)
            else:
                struct_models[s] = list(models)
        else:
            seq_only_models.extend(models)

    # Make grouping model that is parent of chain models.
    mlist = []
    nmodels = 0
    for structure, models in struct_models.items():
        from chimerax.core.models import Model
        group = Model('%s AlphaFold' % structure.name, structure.session)
        group.added_to_session = lambda session, g=group: _log_alphafold_chain_info(g)
        group.add(models)
        mlist.append(group)
        nmodels += len(models)

    mlist.extend(seq_only_models)
    nmodels += len(seq_only_models)
    
    return mlist, nmodels

def _log_alphafold_chain_info(alphafold_group_model):
    am = alphafold_group_model
    struct_name = am.name.rstrip(' AlphaFold')
    from chimerax.core.logger import html_table_params
    lines = ['<table %s>' % html_table_params,
             '  <thead>',
             '    <tr><th colspan=7>AlphaFold chains matching %s</th>' % struct_name,
             '    <tr><th>Chain<th>UniProt Name<th>UniProt Id<th>RMSD<th>Length<th>Seen<th>% Id',
             '  </thead>',
             '  <tbody>',
        ]

    rows = []
    for m in am.child_models():
        cid = ' '.join(_sel_chain_cmd(m,c.chain_id) for c in m.chains)
        rmsd = ('%.2f' % m.rmsd) if hasattr(m, 'rmsd') else ''
        pct_id = '%.0f' % (100*m.seq_identity) if hasattr(m, 'seq_identity') else 'N/A'
        rows.append((cid, m.uniprot_name, m.uniprot_id, rmsd,
                     m.num_residues, m.observed_num_res, pct_id))

    # Combine rows that are identical except chain id.
    row_cids = {}
    for row in rows:
        values = row[1:]
        if values in row_cids:
            row_cids[values].append(row[0])
        else:
            row_cids[values] = [row[0]]
    urows = [(' '.join(cids),) + values for values, cids in row_cids.items()]
    urows.sort(key = lambda row: row[1])	# Sort by UniProt name
    for urow in urows:
        lines.extend([
            '    <tr>',
            '\n'.join('      <td style="text-align:center">%s' % field for field in urow)])
    lines.extend(['  </tbody>',
                  '</table>'])
    msg = '\n'.join(lines)
    
    m.session.logger.info(msg, is_html = True)

def _sel_chain_cmd(structure, chain_id):
    spec = '#%s/%s' % (structure.id_string, chain_id)
    return '<a title="Select chain" href="cxcmd:select %s">%s</a>' % (spec, chain_id)
    
def register_alphafold_match_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg
    from .seqarg import SequencesArg
    desc = CmdDesc(
        required = [('sequences', SequencesArg)],
        keyword = [('color_confidence', BoolArg),
                   ('trim', BoolArg),
                   ('search', BoolArg),
                   ('ignore_cache', BoolArg)],
        synopsis = 'Fetch AlphaFold database models matching an open structure'
    )
    register('alphafold match', desc, alphafold_match, logger=logger)
