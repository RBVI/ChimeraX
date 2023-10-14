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

# -----------------------------------------------------------------------------
# Fetch structures from EBI AlphaFold database using UniProt sequence ID.
# Example for UniProt P29474.
#
#	https://alphafold.ebi.ac.uk/files/AF-P29474-F1-model_v1.cif
#
def alphafold_fetch(session, uniprot_id, color_confidence=True,
                    align_to=None, trim=True, pae=False, ignore_cache=False,
                    add_to_session=True, version=None, in_file_history=True, **kw):

    uniprot_name = uniprot_id if '_' in uniprot_id else None
    uniprot_id = _parse_uniprot_id(uniprot_id)
    from . import database
    url = database.alphafold_model_url(session, uniprot_id, version)
    file_name = url.split('/')[-1]
    
    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, url, 'AlphaFold %s' % uniprot_id, file_name, 'AlphaFold',
                          ignore_cache=ignore_cache, error_status = False)

    model_name = 'AlphaFold %s' % (uniprot_name or uniprot_id)
    models, status = session.open_command.open_data(filename, format = 'mmCIF',
                                                    name = model_name,
                                                    in_file_history = in_file_history,
                                                    **kw)

    from .search import DatabaseEntryId
    db_id = DatabaseEntryId(uniprot_id, name = uniprot_name, version=version)
    from .match import _set_model_database_info
    _set_model_database_info(models, db_id)

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
        from .pae import alphafold_pae
        alphafold_pae(session, structure = models[0], uniprot_id = uniprot_id, version = version)
        
    return models, status

def _parse_uniprot_id(uniprot_id):
    from chimerax.core.errors import UserError
    if '_' in uniprot_id:
        from chimerax.uniprot import map_uniprot_ident
        try:
            return map_uniprot_ident(uniprot_id, return_value = 'entry')
        except Exception:
            raise UserError('UniProt name "%s" not found' % uniprot_id)
    if len(uniprot_id) not in (6, 10):
        raise UserError('UniProt identifiers must be 6 or 10 characters long, got "%s"'
                        % uniprot_id)
    return uniprot_id.upper()

def _color_by_confidence(structure, palette_name = 'alphafold'):
    from chimerax.core.colors import BuiltinColormaps
    from chimerax.std_commands.color import color_by_attr
    color_by_attr(structure.session, 'bfactor', atoms = structure.atoms,
                  palette = BuiltinColormaps[palette_name], log_info = False)

def _align_and_trim(models, align_to_chain, trim):
    from . import match
    for alphafold_model in models:
        match._rename_chains(alphafold_model, [align_to_chain.chain_id])
        match._align_to_chain(alphafold_model, align_to_chain)
        if trim:
            seq_range = getattr(alphafold_model, 'seq_match_range', None)
            if seq_range:
                match._trim_sequence(alphafold_model, seq_range)

def _log_chain_info(models, align_to_name, prediction_method = None):
    for m in models:
        def _show_chain_table(session, m=m):
            from chimerax.atomic import AtomicStructure
            AtomicStructure.added_to_session(m, session)
            from .match import _log_chain_table
            _log_chain_table([m], align_to_name, prediction_method=prediction_method)
        m.added_to_session = _show_chain_table
        m._log_info = False   # Don't show standard chain table

def register_alphafold_fetch_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, StringArg, IntArg
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [('uniprot_id', StringArg)],
        keyword = [('color_confidence', BoolArg),
                   ('align_to', ChainArg),
                   ('trim', BoolArg),
                   ('pae', BoolArg),
                   ('ignore_cache', BoolArg),
                   ('version', IntArg)],
        synopsis = 'Fetch AlphaFold database models for a UniProt identifier'
    )
    
    register('alphafold fetch', desc, alphafold_fetch, logger=logger)
