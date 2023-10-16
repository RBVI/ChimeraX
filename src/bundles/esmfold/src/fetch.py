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
# Fetch structures from ESM Metagenomic Atlas by MGnify accession code.
# Example for MGnify MGYP002537940442
#
#	https://api.esmatlas.com/fetchPredictedStructure/MGYP002537940442
#
def esmfold_fetch(session, mgnify_id, color_confidence=True,
                  align_to=None, trim=True, pae=False, ignore_cache=False,
                  add_to_session=True, version=None, in_file_history=True, **kw):

    _validate_mgnify_id(mgnify_id)

    from .database import esmfold_model_url
    url, file_name = esmfold_model_url(session, mgnify_id, database_version = version)
    
    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, url, 'ESM Metagenomics Atlas %s' % mgnify_id,
                          file_name, 'ESMFold',
                          ignore_cache=ignore_cache, error_status = False)

    model_name = 'ESMFold %s' % mgnify_id
    models, status = session.open_command.open_data(filename, format = 'PDB',
                                                    name = model_name,
                                                    in_file_history = in_file_history,
                                                    **kw)

    from .search import DatabaseEntryId
    db_id = DatabaseEntryId(mgnify_id, id_type = 'MGnify', database = 'ESMFold', version=version)
    from chimerax.alphafold.match import _set_model_database_info
    _set_model_database_info(models, db_id)

    if color_confidence:
        for s in models:
            # Set initial style so confidence coloring is not replaced.
            s.apply_auto_styling()
            s._auto_style = False
            _color_by_confidence(s, palette_name = 'esmfold')

    if align_to is not None:
        _align_and_trim(models, align_to, trim)
        _log_chain_info(models, align_to.name)
        
    if add_to_session:
        session.models.add(models)

    if pae:
        from .pae import esmfold_pae
        esmfold_pae(session, structure = models[0], mgnify_id = mgnify_id, version = version)
        
    return models, status

def _validate_mgnify_id(mgnify_id):
    from chimerax.core.errors import UserError
    if not mgnify_id.startswith('MGYP'):
        raise UserError(f'MGnify identifier must start with MGYP, got "{mgnify_id}"')
    if len(mgnify_id) != 16 or not mgnify_id[4:16].isdigit():
        raise UserError(f'MGnify identifier must have 12 digits after MGYP, got "{mgnify_id}"')

from chimerax.alphafold.fetch import _color_by_confidence, _align_and_trim

def _log_chain_info(models, align_to_name):
    for m in models:
        def _show_chain_table(session, m=m):
            from chimerax.atomic import AtomicStructure
            AtomicStructure.added_to_session(m, session)
            from chimerax.alphafold.match import _log_chain_table
            _log_chain_table([m], align_to_name)
        m.added_to_session = _show_chain_table
        m._log_info = False   # Don't show standard chain table

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
