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

def alphafold_match(session, chains, color_confidence=True, trim = True,
                    search = True, ignore_cache=False):
    from chimerax.atomic import Residue
    chains = [chain for chain in chains if chain.polymer_type == Residue.PT_AMINO]
    if len(chains) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No protein chains specified')

    log = session.logger

    # Use UniProt identifiers in file metadata to get AlphaFold models.
    chain_models = _fetch_by_uniprot_id(chains, color_confidence=color_confidence,
                                        trim=trim, local=(search == 'local'), log=log,
                                        ignore_cache=ignore_cache)
        
    # Try sequence search if some chains were not found by UniProt identifier.
    if search:
        search_chains = [chain for chain in chains if chain not in chain_models]
        search_chain_models = _fetch_by_sequence(search_chains, color_confidence=color_confidence,
                                                 trim=trim, local=(search == 'local'), log=log,
                                                 ignore_cache=ignore_cache)
        chain_models.update(search_chain_models)

    # Report chains with no AlphaFold model
    chains_no_model = [chain for chain in chains if chain not in chain_models]
    if chains_no_model:
        cnames = ','.join(c.chain_id for c in chains_no_model)
        msg = ('No matching AlphaFold model for chain%s %s'
               % (_plural(chains_no_model), cnames))
        log.warning(msg)

    mlist, nchains = _group_chains_by_structure(chain_models)
    session.models.add(mlist)

    msg = 'Opened %d AlphaFold chain model%s' % (nchains, _plural(nchains))
    session.logger.info(msg)

    return mlist

def _fetch_by_uniprot_id(chains, color_confidence = True, trim = True,
                         local = False, ignore_cache = False, log = None):
    # Get Uniprot ids from mmCIF or PDB file metadata
    chain_uids = _chain_uniprot_ids(chains)
    chain_models, missing_uids = _alphafold_models(chains, chain_uids,
                                                   color_confidence=color_confidence, trim=trim,
                                                   ignore_cache=ignore_cache)
    if missing_uids:
        log.warning('Structure metadata included %d UniProt id%s %s'
                    % (len(missing_uids), _plural(missing_uids), _uniprot_chain_info(missing_uids)) +
                    (' that do not have AlphaFold database models.' if len(missing_uids) > 1 else
                     ' that does not have an AlphaFold database model.'))
    if chain_models:
        uid_chains = _uniprot_chains(chain_models)
        log.info('%d AlphaFold model%s found using UniProt identifier%s %s from structure file metadata'
                 % (len(uid_chains), _plural(uid_chains), _plural(uid_chains),
                    _uniprot_chain_info(uid_chains)))
    return chain_models

def _fetch_by_sequence(chains, color_confidence = True, trim = True,
                       local = False, ignore_cache = False, log = None):
    from .search import chain_sequence_search, SearchError
    try:
        chain_uids = chain_sequence_search(chains, local = local)
    except SearchError as e:
        log.error(str(e))
        return {}
    
    chain_models, missing_uids = \
        _alphafold_models(chains, chain_uids,
                          color_confidence=color_confidence, trim=trim,
                          ignore_cache=ignore_cache)
    if missing_uids and log:
        missing_names = ', '.join('%s (chains %s)' % (uid,','.join(cnames))
                                  for uid,cnames in missing_uids.items())
        log.warning('Sequence search found %d UniProt id%s %s'
                    % (len(missing_uids), _plural(missing_uids), missing_names) +
                    ' that do not have AlphaFold database models.')
    if chain_models and log:
        uid_chains = _uniprot_chains(chain_models)
        log.info('%d AlphaFold model%s %s found using sequence similarity searches'
                 % (len(uid_chains), _plural(uid_chains), _uniprot_chain_info(uid_chains)))

    return chain_models

def _alphafold_models(chains, chain_uids, color_confidence=True, trim=True,
                      ignore_cache=False):
    chain_models = {}
    missing = {}
    from chimerax.core.errors import UserError
    from .fetch import alphafold_fetch
    for chain, uid in chain_uids:
        if uid.uniprot_id in missing:
            missing[uid.uniprot_id].append(chain.chain_id)
            continue
        session = chain.structure.session
        try:
            models, status = alphafold_fetch(session, uid.uniprot_id,
                                             color_confidence=color_confidence,
                                             add_to_session=False,
                                             ignore_cache=ignore_cache)
        except UserError as e:
            if not str(e).endswith('Not Found'):
                session.logger.warning(str(e))
            missing[uid.uniprot_id] = [chain.chain_id]
            models = []
        for alphafold_model in models:
            alphafold_model._log_info = False          # Don't log chain tables
            alphafold_model.uniprot_id = uid.uniprot_id
            alphafold_model.uniprot_name = uid.uniprot_name
            alphafold_model.observed_num_res = chain.num_existing_residues
            if trim:
                _trim_sequence(alphafold_model, uid.database_sequence_range)
            _rename_chains(alphafold_model, chain)
            _align_to_chain(alphafold_model, chain)
            alphafold_model.name = 'UniProt %s chain %s' % (uid.uniprot_id, chain.chain_id)
        if models:
            if chain in chain_models:
               chain_models[chain].extend(models)
            else:
                chain_models[chain] = models

    return chain_models, missing

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
            structure.seq_identity = _sequence_identity(rseq, mseq)

def _sequence_identity(seq1, seq2):
    m = 0
    for r1, r2 in zip(seq1.characters, seq2.characters):
        if r1 == r2 and r1 != '.':
            m += 1
    d = min(seq1.num_residues, seq2.num_residues)
    return m/d if d > 0 else 0.0

def _chain_uniprot_ids(chains):
    chain_uids = []
    from chimerax.atomic import uniprot_ids
    for structure, schains in _chains_by_structure(chains).items():
        id_to_chain = {chain.chain_id:chain for chain in schains}
        for u in uniprot_ids(structure):
            chain = id_to_chain.get(u.chain_id)
            if chain:
                chain_uids.append((chain, u))
    return chain_uids

def _chains_by_structure(chains):
    struct_chains = {}
    for chain in chains:
        s = chain.structure
        if s in struct_chains:
            struct_chains[s].append(chain)
        else:
            struct_chains[s] = [chain]
    return struct_chains

def _uniprot_chain_info(uniprot_chains):
    info = ', '.join('%s (chain%s %s)' % (uid, ('s' if len(cnames) > 1 else ''), ','.join(cnames))
                     for uid,cnames in uniprot_chains.items())
    return info

def _uniprot_chains(chain_models):
    uc = {}
    for chain, models in chain_models.items():
        for model in models:
            uid = model.uniprot_id
            if uid in uc:
                uc[uid].append(chain.chain_id)
            else:
                uc[uid] = [chain.chain_id]
    return uc

def _plural(seq):
    n = seq if isinstance(seq, int) else len(seq)
    return 's' if n > 1 else ''

def _group_chains_by_structure(chain_models):
    
    # Group models by structure
    struct_models = {}
    for chain,models in chain_models.items():
        s = chain.structure
        if s in struct_models:
            struct_models[s].extend(models)
        else:
            struct_models[s] = list(models)

    # Make grouping model that is parent of chain models.
    mlist = []
    nchains = 0
    for structure, models in struct_models.items():
        from chimerax.core.models import Model
        group = Model('%s AlphaFold' % structure.name, structure.session)
        group.added_to_session = lambda session, g=group: _log_alphafold_chain_info(g)
        group.add(models)
        mlist.append(group)
        nchains += len(models)

    return mlist, nchains

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
    msorted = list(am.child_models())
    msorted.sort(key = lambda m: m.chains[0].chain_id if m.chains else '')
    for m in msorted:
        cid = ', '.join(_sel_chain_cmd(m,c.chain_id) for c in m.chains)
        rmsd = ('%.2f' % m.rmsd) if hasattr(m, 'rmsd') else ''
        pct_id = '%.0f' % (100*m.seq_identity) if hasattr(m, 'seq_identity') else 'N/A'
        lines.extend([
            '    <tr>',
            '      <td style="text-align:center">%s' % cid,
            '      <td style="text-align:center">%s' % m.uniprot_name,
            '      <td style="text-align:center">%s' % m.uniprot_id,
            '      <td style="text-align:center">%s' % rmsd,
            '      <td style="text-align:center">%s' % m.num_residues,
            '      <td style="text-align:center">%s' % m.observed_num_res,
            '      <td style="text-align:center">%s' % pct_id])
    lines.extend(['  </tbody>',
                  '</table>'])
    msg = '\n'.join(lines)
    
    m.session.logger.info(msg, is_html = True)

def _sel_chain_cmd(structure, chain_id):
    spec = '#%s/%s' % (structure.id_string, chain_id)
    return '<a title="Select chain" href="cxcmd:select %s">%s</a>' % (spec, chain_id)
    
def register_alphafold_match_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg
    from chimerax.atomic import UniqueChainsArg
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        keyword = [('color_confidence', BoolArg),
                   ('trim', BoolArg),
                   ('search', BoolArg),
                   ('ignore_cache', BoolArg)],
        synopsis = 'Fetch AlphaFold database models matching an open structure'
    )
    register('alphafold match', desc, alphafold_match, logger=logger)
