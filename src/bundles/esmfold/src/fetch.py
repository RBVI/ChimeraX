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
# Fetch structures from ESM Metagenomic Atlas by MGnify accession code.
# Example for MGnify MGYP002537940442
#
#	https://api.esmatlas.com/fetchPredictedStructure/MGYP002537940442
#
esmfold_fetch_url = 'https://api.esmatlas.com/fetchPredictedStructure/'
def esmfold_fetch(session, mgnify_id, color_confidence=True,
                    align_to=None, trim=True, pae=False, ignore_cache=False,
                    add_to_session=True, version=None, in_file_history=True, **kw):

    url = esmfold_fetch_url + mgnify_id
    file_name = mgnify_id + '.pdb'
    
    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, url, 'ESM Metagenomics Atlas %s' % mgnify_id,
                          file_name, 'ESMFold',
                          ignore_cache=ignore_cache, error_status = False)

    model_name = 'ESMFold %s' % mgnify_id
    models, status = session.open_command.open_data(filename, format = 'PDB',
                                                    name = model_name,
                                                    in_file_history = in_file_history,
                                                    **kw)
    for m in models:
        m.mgnify_id = mgnify_id

    if color_confidence:
        for s in models:
            # Set initial style so confidence coloring is not replaced.
            s.apply_auto_styling()
            s._auto_style = False
            _color_by_confidence(s)

    if align_to is not None:
        _align_and_trim(models, align_to, trim)
        _log_chain_info(models, align_to.name)
        
    if add_to_session:
        session.models.add(models)

    if pae:
        from .pae import esmfold_pae
        esmfold_pae(session, structure = models[0], mgnify_id = mgnify_id, version = version)
        
    return models, status

def _color_by_confidence(structure):
    from chimerax.core.colors import BuiltinColormaps
    from chimerax.std_commands.color import color_by_attr
    color_by_attr(structure.session, 'bfactor', atoms = structure.atoms,
                  palette = BuiltinColormaps['esmfold'],
                  log_info = False)

def _align_and_trim(models, align_to_chain, trim):
    from chimerax.alphafold import match
    for esmfold_model in models:
        match._rename_chains(esmfold_model, [align_to_chain.chain_id])
        match._align_to_chain(esmfold_model, align_to_chain)
        if trim:
            seq_range = getattr(esmfold_model, 'seq_match_range', None)
            if seq_range:
                match._trim_sequence(esmfold_model, seq_range)

def _log_chain_info(models, align_to_name):
    for m in models:
        def _show_chain_table(session, m=m):
            from chimerax.atomic import AtomicStructure
            AtomicStructure.added_to_session(m, session)
            _log_esmfold_chain_table([m], align_to_name)
        m.added_to_session = _show_chain_table
        m._log_info = False   # Don't show standard chain table

def _log_esmfold_chain_table(chain_models, match_to_name):
    from chimerax.core.logger import html_table_params
    lines = ['<table %s>' % html_table_params,
             '  <thead>',
             '    <tr><th colspan=7>ESMFold chains matching %s</th>' % match_to_name,
             '    <tr><th>Chain<th>MGnify Id<th>RMSD<th>Length<th>Seen<th>% Id',
             '  </thead>',
             '  <tbody>',
    ]

    rows = []
    for m in chain_models:
        cid = ' '.join(_sel_chain_cmd(m,c.chain_id) for c in m.chains)
        rmsd = ('%.2f' % m.rmsd) if hasattr(m, 'rmsd') else ''
        pct_id = '%.0f' % (100*m.seq_identity) if hasattr(m, 'seq_identity') else 'N/A'
        mgid = getattr(m, 'mgnify_id', '')
        rows.append((cid, mgid, rmsd,
                     m.num_residues, m.num_observed_residues, pct_id))

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

def register_esmfold_fetch_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, StringArg, IntArg
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [('mgnify_id', StringArg)],
        keyword = [('color_confidence', BoolArg),
                   ('align_to', ChainArg),
                   ('trim', BoolArg),
                   ('pae', BoolArg),
                   ('ignore_cache', BoolArg),
                   ('version', IntArg)],
        synopsis = 'Fetch ESM Metagenomic Atlas database model for a MGnify identifier'
    )
    
    register('esmfold fetch', desc, esmfold_fetch, logger=logger)
