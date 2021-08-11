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
def alphafold_fetch(session, uniprot_id, color_confidence=True,
                    ignore_cache=False, add_to_session=True, **kw):

    from chimerax.core.errors import UserError
    if len(uniprot_id) not in (6, 10):
        raise UserError('UniProt identifiers must be 6 or 10 characters long, got "%s"'
                        % uniprot_id)

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

    if add_to_session:
        session.models.add(models)
        
    return models, status

def _color_by_confidence(structure):
    from chimerax.core.colors import Colormap, BuiltinColors
    colors = [BuiltinColors[name] for name in ('red', 'orange', 'yellow', 'cornflowerblue', 'blue')]
    palette = Colormap((0, 50, 70, 90, 100), colors)
    from chimerax.std_commands.color import color_by_attr
    color_by_attr(structure.session, 'bfactor', atoms = structure.atoms, palette = palette,
                  log_info = False)
    
def register_alphafold_fetch_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, StringArg
    desc = CmdDesc(
        required = [('uniprot_id', StringArg)],
        keyword = [('color_confidence', BoolArg),
                   ('ignore_cache', BoolArg)],
        synopsis = 'Fetch AlphaFold database models for a UniProt identifier'
    )
    
    register('alphafold fetch', desc, alphafold_fetch, logger=logger)
