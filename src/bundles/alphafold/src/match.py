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

def alphafold_match(session, sequences, color_confidence=True, trim = True,
                    search = True, pae = False, ignore_cache=False):
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
                                      trim=trim, log=log, ignore_cache=ignore_cache)

    # Try sequence search if some sequences were not found by UniProt identifier.
    if search:
        from .fetch import alphafold_fetch
        from .search import alphafold_sequence_search
        search_seqs = [seq for seq in sequences if seq not in seq_models]
        search_seq_models = _fetch_by_sequence(session, search_seqs,
                                               color_confidence=color_confidence, trim=trim,
                                               sequence_search=alphafold_sequence_search,
                                               fetch=alphafold_fetch, ignore_cache=ignore_cache, log=log)
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

    if pae:
        for seq,models in seq_models.items():
            for m in models:
                from .pae import alphafold_pae
                alphafold_pae(session, structure = m, uniprot_id = m.database.id,
                              version = m.database.version)

    return mlist

def _fetch_by_uniprot_id(session, sequences, color_confidence = True, trim = True,
                         ignore_cache = False, log = None):
    # Get Uniprot ids from mmCIF or PDB file metadata
    sequence_uids = _sequence_uniprot_ids(sequences)
    from .fetch import alphafold_fetch
    seq_models, missing_uids = _database_models(session, sequence_uids,
                                                color_confidence=color_confidence, trim=trim,
                                                fetch=alphafold_fetch, ignore_cache=ignore_cache)
    if missing_uids:
        log.warning('%d UniProt id%s'
                    % (len(missing_uids), _plural(missing_uids)) +
                    (' do not have AlphaFold database models: ' if len(missing_uids) > 1 else
                     ' does not have an AlphaFold database model: ') +
                    _id_and_sequences_description(missing_uids))
    if seq_models:
        uid_seqs = _sequences_by_id(seq_models)
        log.info('%d AlphaFold model%s found using UniProt identifier%s: %s'
                 % (len(uid_seqs), _plural(uid_seqs), _plural(uid_seqs),
                    _id_and_sequences_description(uid_seqs)))
    return seq_models

def _fetch_by_sequence(session, sequences, color_confidence = True, trim = True,
                       sequence_search = None, fetch = None, ignore_cache = False, log = None):

    from .search import SearchError
    seq_strings = [seq.characters for seq in sequences]
    if [seq for seq in seq_strings if len(seq) < 20]:
        log.warning('Cannot search short sequences less than 20 amino acids')
    try:
        match_uids = sequence_search(seq_strings, log = log)
    except SearchError as e:
        log.error(str(e))
        return {}

    seq_uids = [(seq,uid) for seq, uid in zip(sequences, match_uids) if uid is not None]
    from chimerax.atomic import Chain
    for seq, uid in seq_uids:
        if isinstance(seq, Chain):
            uid.chain_id = seq.chain_id

    seq_models, missing_uids = \
        _database_models(session, seq_uids,
                         color_confidence=color_confidence, trim=trim,
                         fetch=fetch, ignore_cache=ignore_cache)

    if missing_uids and log:
        missing_names = ', '.join('%s (%s)' % (uid, _sequences_description(seqs))
                                  for uid,seqs in missing_uids.items())
        db_name = seg_uids[0][1].database
        log.warning('Sequence search found %d %s id%s'
                    % (len(missing_uids), db_name, _plural(missing_uids)) +
                    ' that do not have %s database models: %s' % (db_name, missing_names))
    if seq_models and log:
        uid_seqs = _sequences_by_id(seq_models)
        db_name = seq_uids[0][1].database
        log.info('%d %s model%s found using sequence similarity searches: %s'
                 % (len(uid_seqs), db_name, _plural(uid_seqs), _id_and_sequences_description(uid_seqs)))

    return seq_models

def _database_models(session, sequence_database_ids, color_confidence=True, trim=True,
                     fetch=None, ignore_cache=False):
    seq_models = {}
    missing = {}
    from chimerax.core.errors import UserError
    from chimerax.atomic import Chain
    for seq, db_id in sequence_database_ids:
        if db_id.id in missing:
            missing[db_id.id].append(seq)
            continue
        try:
            models, status = fetch(session, db_id.id,
                                   version = db_id.version,
                                   color_confidence=color_confidence,
                                   add_to_session=False,
                                   in_file_history=(len(sequence_database_ids)==1),
                                   ignore_cache=ignore_cache)
        except UserError as e:
            if not str(e).endswith('Not Found'):
                session.logger.warning(str(e))
            missing[db_id.id] = [seq]
            models = []
        _set_model_database_info(models, db_id)
        for database_model in models:
            database_model._log_info = False          # Don't log chain tables
            seq_range = getattr(db_id, 'database_sequence_range', None)
            if trim and seq_range:
                _trim_sequence(database_model, seq_range)
            if isinstance(seq, Chain):
                _rename_chains(database_model, [seq.chain_id])
                _align_to_chain(database_model, seq, use_dssp = False)
            else:
                _log_sequence_similarity(database_model, seq)
            if trim and seq_range is None:
                seq_range = getattr(database_model, 'seq_match_range', None)
                if seq_range:
                    _trim_sequence(database_model, seq_range)
            uname = db_id.name or db_id.id
            database_model.name = f'{db_id.database} {uname}'
            if isinstance(seq, Chain):
                database_model.name += ' chain %s' % seq.chain_id
        if models:
            if seq in seq_models:
                seq_models[seq].extend(models)
            else:
                seq_models[seq] = models

    return seq_models, missing

def _set_model_database_info(models, database_entry_id):
    for model in models:
        model.database = database_entry_id
        # Save attribute in sessions
        from chimerax.atomic import AtomicStructure
        AtomicStructure.register_attr(model.session, 'database', database_entry_id.database)

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

def _rename_chains(structure, chain_ids):
    schains = structure.chains
    if len(schains) != len(chain_ids):
        cnames = ', '.join(c.chain_id for c in schains)
        structure.session.logger.warning('Alphafold structure %s has %d chains (%s), expected %d.'
                                         % (structure.name, len(schains), cnames, len(chain_ids)) +
                                         '  Not renaming chain id to match target structure.')

    for schain, chain_id in zip(schains, chain_ids):
        schain.chain_id = chain_id

def _align_to_chain(structure, chain, use_dssp = True):

    from chimerax.match_maker.match import cmd_match
    results = cmd_match(structure.session, structure.atoms, to = chain.existing_residues.atoms,
                        compute_s_s = use_dssp, verbose=None)

    if structure.num_chains > 1:
        return  # Don't add alignment attributes to multi-chain structures

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
            _set_match_attributes(rseq, mseq)

    structure.num_observed_residues = chain.num_existing_residues

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

def _sequence_coverage(seq1, seq2):
    l2 = c = 0
    for aa1, aa2 in zip(seq1.characters, seq2.characters):
        if aa2 != '.':
            l2 += 1
            if aa1 != '.':
                c += 1
    return c/l2 if l2 > 0 else 0.0

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

def _set_match_attributes(expt_seq, alpha_seq):
    '''
    Set sequence_match, missing_structure and c_alpha_distance residue attributes.
    '''
    for c1, r1, c2, r2 in _paired_residues(expt_seq, alpha_seq):
        if r1:
            r1.same_sequence = (c1 == c2)
        if r2:
            r2.same_sequence = (c1 == c2)
            r2.missing_structure = (r1 is None)
        d = _c_alpha_distance(r1, r2)
        if d is not None:
            r1.c_alpha_distance = r2.c_alpha_distance = d

    # Make attributes save in sessions
    session = expt_seq.structure.session
    from chimerax.atomic import Residue
    Residue.register_attr(session, "same_sequence", "AlphaFold", attr_type=bool)
    Residue.register_attr(session, "missing_structure", "AlphaFold", attr_type=bool)
    Residue.register_attr(session, "c_alpha_distance", "AlphaFold", attr_type=float)

def _paired_residues(seq1, seq2):
    c1, c2 = seq1.characters, seq2.characters
    r1 = {seq1.ungapped_to_gapped(i):r for i,r in enumerate(seq1.residues)}
    r2 = {seq2.ungapped_to_gapped(i):r for i,r in enumerate(seq2.residues)}
    pairs = [(c1, r1.get(i), c2, r2.get(i)) for i,(c1,c2) in enumerate(zip(c1,c2))]
    return pairs

def _c_alpha_distance(r1, r2):
    if r1 is None or r2 is None:
        return None
    ca1, ca2 = r1.find_atom("CA"), r2.find_atom("CA")
    if ca1 is None or ca2 is None:
        return None
    from chimerax.geometry import distance
    d = distance(ca1.scene_coord, ca2.scene_coord)
    return d

def _set_same_sequence_attribute(chain_seq, seq):
    c1, c2 = chain_seq.characters, seq.characters
    for i, r in enumerate(chain_seq.residues):
        ci = chain_seq.ungapped_to_gapped(i)
        r.same_sequence = (c1[ci] == c2[ci])
    session = chain_seq.structure.session
    from chimerax.atomic import Residue
    Residue.register_attr(session, "same_sequence", "AlphaFold", attr_type=bool)

def _sequence_uniprot_ids(sequences):
    seq_uids = []
    from chimerax.atomic import uniprot_ids
    for structure, schains in _chains_by_structure(sequences).items():
        id_to_chain = {chain.chain_id:chain for chain in schains}
        for u in uniprot_ids(structure):
            chain = id_to_chain.get(u.chain_id)
            if chain:
                from .search import DatabaseEntryId
                uid = DatabaseEntryId(u.uniprot_id, name = u.uniprot_name)
                uid.database_sequence_range = u.database_sequence_range
                seq_uids.append((chain, uid))
    for seq in sequences:
        if hasattr(seq, 'uniprot_accession'):
            from .search import DatabaseEntryId
            uid = DatabaseEntryId(seq.uniprot_accession, name = getattr(seq, 'uniprot_name', None))
            seq_uids.append((seq, uid))
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

def _id_and_sequences_description(id_sequences):
    info = ', '.join('%s (%s)' % (id, _sequences_description(seqs))
                     for id,seqs in id_sequences.items())
    return info

def _sequences_by_id(seq_models):
    id_to_seqs = {}
    for seq, models in seq_models.items():
        for model in models:
            uid = model.database.id
            if uid in id_to_seqs:
                id_to_seqs[uid].append(seq)
            else:
                id_to_seqs[uid] = [seq]
    return id_to_seqs

def _sequences_description(seqs):
    from chimerax.atomic import Chain
    cids, uids, mgids, nseq = [], [], [], 0
    for seq in seqs:
        if isinstance(seq, Chain):
            cids.append(seq.chain_id)
        elif hasattr(seq, 'uniprot_name'):
            uids.append(seq.uniprot_name)
        elif hasattr(seq, 'uniprot_accession'):
            uids.append(seq.uniprot_accession)
        elif hasattr(seq, 'mgnify_accession'):
            mgids.append(seq.mgnify_accession)
        else:
            nseq += 1
    items = []
    if cids:
        items.append('chain%s %s' % (_plural(cids), ','.join(cids)))
    if uids:
        items.append('UniProt %s' % ', '.join(uids))
    if mgids:
        items.append('MGnify %s' % ', '.join(mgids))
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
        database_name = models[0].database.database	# 'AlphaFold'
        from chimerax.core.models import Model
        group = Model(f'{structure.name} {database_name}', structure.session)
        group.added_to_session = lambda session, g=group: _log_chain_info(g)
        group.add(models)
        mlist.append(group)
        nmodels += len(models)

    mlist.extend(seq_only_models)
    nmodels += len(seq_only_models)

    return mlist, nmodels

def _log_chain_info(group_model):
    models = group_model.child_models()
    if models:
        database_name = models[0].database.database
        struct_name = group_model.name.removesuffix(' ' + database_name)
        _log_chain_table(models, struct_name)

def _log_chain_table(chain_models, match_to_name, prediction_method = None):
    db = getattr(chain_models[0], 'database', None)
    if db:
        db_name = db.database
        id_titles = f'<th>{db.id_type} Id'
        use_db_name = (db.name is not None)
        if use_db_name:
            id_titles += f'<th>{db.id_type} Name'
    else:
        db_name = prediction_method if prediction_method else ''
        id_titles = ''


    from chimerax.core.logger import html_table_params
    lines = [f'<table {html_table_params}>',
             '  <thead>',
             f'    <tr><th colspan=7>{db_name} prediction matching {match_to_name}</th>',
             f'    <tr><th>Chain{id_titles}<th>RMSD<th>Length<th>Seen<th>% Id',
             '  </thead>',
             '  <tbody>',
    ]

    rows = []
    for m in chain_models:
        cid = ' '.join(_sel_chain_cmd(m,c.chain_id) for c in m.chains)
        rmsd = ('%.2f' % m.rmsd) if hasattr(m, 'rmsd') else ''
        pct_id = '%.0f' % (100*m.seq_identity) if hasattr(m, 'seq_identity') else 'N/A'
        if db:
            ids = (m.database.id, m.database.name) if use_db_name else (m.database.id,)
        else:
            ids = ()
        rows.append((cid,) + ids + (rmsd, m.num_residues, m.num_observed_residues, pct_id))

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

def _log_sequence_similarity(alphafold_model, seq, database_name = 'AlphaFold'):
    if alphafold_model.num_chains != 1:
        return
    def _log_info(session, am=alphafold_model, seq=seq):
        mname = am.name.removeprefix(database_name + ' ')
        msg = _similarity_table_html(am.chains[0], seq, mname, database_name = database_name)
        am.session.logger.info(msg, is_html = True)
        from chimerax.atomic import AtomicStructure
        AtomicStructure.added_to_session(alphafold_model, session)
    alphafold_model.added_to_session = _log_info

def _similarity_table_html(mseq, seq, mname, database_name = 'AlphaFold'):
    from chimerax.alignment_algs.NeedlemanWunsch import nw
    score, (mseq_gapped, seq_gapped) = nw(mseq, seq, return_seqs = True)
    _set_same_sequence_attribute(mseq_gapped, seq_gapped)
    si = _sequence_identity(mseq_gapped, seq_gapped)
    sc = _sequence_coverage(mseq_gapped, seq_gapped)
    fields = [mname, _sequence_name(seq), '%.1f' % (100*si), '%.1f' % (100*sc)]
    from chimerax.core.logger import html_table_params
    lines = [f'<table {html_table_params}>',
             '  <thead>',
             '    <tr><th colspan=4>Sequence Similarity',
             f'    <tr><th>{database_name} Model<th>Query Sequence<th>Identity %<th>Coverage %',
             '  </thead>',
             '  <tbody>',
             '\n'.join('    <td style="text-align:center">%s' % field for field in fields),
             '  </tbody>',
             '</table>']
    msg = '\n'.join(lines)
    return msg

def _sequence_name(seq):
    if seq.name and seq.name != 'query' and seq.name != 'sequence':
        return seq.name
    if hasattr(seq, 'uniprot_name'):
        return seq.uniprot_name
    if hasattr(seq, 'uniprot_accession'):
        return seq.uniprot_accession
    if hasattr(seq, 'mgnify_accession'):
        return seq.mgnify_accession
    ug = seq.ungapped()
    if len(ug) <= 10:
        return ug
    return ug[:5] + '...' + ug[-5:]

def register_alphafold_match_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg
    from chimerax.atomic import SequencesArg
    desc = CmdDesc(
        required = [('sequences', SequencesArg)],
        keyword = [('color_confidence', BoolArg),
                   ('trim', BoolArg),
                   ('search', BoolArg),
                   ('pae', BoolArg),
                   ('ignore_cache', BoolArg)],
        synopsis = 'Fetch AlphaFold database models matching an open structure'
    )
    register('alphafold match', desc, alphafold_match, logger=logger)
