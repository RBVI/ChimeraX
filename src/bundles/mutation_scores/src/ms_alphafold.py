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


# -----------------------------------------------------------------------------
# Fetch AlphaFold database model for mutation data using UniProt identifier
# or sequence derived from mutations.  If there are sequence mismatches associate
# mutation data with AFDB structure by calculating an alignment.
#
def mutation_scores_alphafold(session, mutation_set = None):
    from .ms_data import mutation_scores
    mset = mutation_scores(session, mutation_set)
    uniprot_id = getattr(mset, 'uniprot_id', None)
    if uniprot_id:
        cmd = f'open {mset.uniprot_id} from alphafold name "{mutation_set} {uniprot_id}"'
    else:
        # Search based on sequence
        seq = mset.sequence().characters
        cmd = f'alphafold match {seq}'

    from chimerax.core.commands import run, quote_if_necessary
    models = run(session, cmd)

    if len(models) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No AlphaFold database model was found.')

    # Check if structure associated. If not try aligning sequences.
    associated = False
    af_chains = models[0].chains
    for c in mset.associated_chains():
        if c in af_chains:
            associated = True
            break
    if not associated:
        cmd = f'mutationscores structure {af_chains[0].atomspec} mutationSet {quote_if_necessary(mset.name)} alignSequences True'
        run(session, cmd)
    
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, FloatArg, BoolArg
    desc = CmdDesc(
        optional = [('mutation_set', StringArg)],
        synopsis = 'Fetch AlphaFold database structure prediction for mutation data sequence.'
    )
    register('mutationscores alphafold', desc, mutation_scores_alphafold, logger=logger)
