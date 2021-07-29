# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
# Fetch structures from EBI AlphaFold database using UniProt sequence ID.
# Example for UniProt P29474.
#
#	https://alphafold.ebi.ac.uk/files/AF-P29474-F1-model_v1.cif
#
def fetch_alphafold(session, uniprot_id, color_confidence=True, trim = True,
                    ignore_cache=False, **kw):

    # Instead of UniProt id can specify an chain specifiers and any uniprot ids
    # associated with those chains will be fetched, trimmed and aligned to the chains.
    chains = _parse_chain_spec(session, uniprot_id)
    if chains:
        return fetch_alphafold_for_chains(session, chains,
                                          color_confidence=color_confidence, trim=trim,
                                          ignore_cache=ignore_cache, **kw)
    
    from chimerax.core.errors import UserError
    if len(uniprot_id) not in (6, 10):
        raise UserError("UniProt identifiers must be 6 or 10 characters long")

    uniprot_id = uniprot_id.upper()
    file_name = 'AF-%s-F1-model_v1.cif' % uniprot_id
    url = 'https://alphafold.ebi.ac.uk/files/' + file_name

    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, url, 'AlphaFold %s' % uniprot_id, file_name, 'AlphaFold',
                          ignore_cache=ignore_cache)

    model_name = 'AlphaFold %s' % uniprot_id
    models, status = session.open_command.open_data(filename, format = 'mmCIF',
                                                    name = model_name, **kw)
    if color_confidence:
        for s in models:
            # Set initial style so confidence coloring is not replaced.
            s.apply_auto_styling()
            s._auto_style = False
            _color_by_confidence(s)
            
    return models, status

def fetch_alphafold_for_chains(session, chains, color_confidence=True, trim=True,
                               ignore_cache=False, **kw):
    chain_uids, chains_no_uid = _chain_uniprot_ids(chains)

    if chains_no_uid:
        msg = ('UniProt sequence identifier not specified in file for chains %s'
               % ','.join(str(c) for c in chains_no_uid))
        session.logger.warning(msg)

    mlist = []
    missing = {}
    naf = 0
    from chimerax.core.errors import UserError
    cbs = _chains_by_structure(chain_uids.keys())
    for structure, schains in cbs.items():
        smlist = []
        for chain in schains:
            uid = chain_uids[chain]
            if uid.uniprot_id in missing:
                missing[uid.uniprot_id].append(chain.chain_id)
                continue
            try:
                models, status = fetch_alphafold(session, uid.uniprot_id, ignore_cache=ignore_cache,
                                                 color_confidence=color_confidence, **kw)
            except UserError as e:
                if not str(e).endswith('Not Found'):
                    session.logger.warning(str(e))
                missing[uid.uniprot_id] = [chain.chain_id]
                models = []
            for alphafold_model in models:
                if trim:
                    _trim_sequence(alphafold_model, uid.chain_sequence_range)
                _rename_chains(alphafold_model, chain)
                _align_to_chain(alphafold_model, chain)
                alphafold_model.name = 'UniProt %s chain %s' % (uid.uniprot_id, chain.chain_id)
            smlist.extend(models)
        if smlist:
            from chimerax.core.models import Model
            group = Model('%s AlphaFold' % structure.name, session)
            group.add(smlist)
            mlist.append(group)
            naf += len(smlist)

    if missing:
        missing_names = ', '.join('%s (chains %s)' % (uid,','.join(cnames))
                                  for uid,cnames in missing.items())
        session.logger.warning('AlphaFold database does not have models for %d UniProt ids %s'
                               % (len(missing), missing_names))

    msg = 'Opened %d AlphaFold chain models' % naf
    return mlist, msg

def _color_by_confidence(structure):
    from chimerax.core.colors import Colormap, BuiltinColors
    colors = [BuiltinColors[name] for name in ('red', 'orange', 'yellow', 'cornflowerblue', 'blue')]
    palette = Colormap((0, 50, 70, 90, 100), colors)
    from chimerax.std_commands.color import color_by_attr
    color_by_attr(structure.session, 'bfactor', atoms = structure.atoms, palette = palette,
                  log_info = False)

def _parse_chain_spec(session, spec):
    from chimerax.atomic import UniqueChainsArg
    try:
        chains, text, rest = UniqueChainsArg.parse(spec, session)
    except Exception:
        return None
    return chains

def _chain_uniprot_ids(chains):
    chain_uids = {}
    chains_no_uid = []
    from chimerax.atomic import uniprot_ids
    for structure, schains in _chains_by_structure(chains).items():
        uids = {u.chain_id:u for u in uniprot_ids(structure)}
        for chain in schains:
            uid = uids.get(chain.chain_id)
            if uid is None:
                chains_no_uid.append(chain)
            else:
                chain_uids[chain] = uid
    return chain_uids, chains_no_uid

def _chains_by_structure(chains):
    struct_chains = {}
    for chain in chains:
        s = chain.structure
        if s in struct_chains:
            struct_chains[s].append(chain)
        else:
            struct_chains[s] = [chain]
    return struct_chains

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
    cmd_match(structure.session, structure.atoms, to = chain.existing_residues.atoms,
              verbose=None)
