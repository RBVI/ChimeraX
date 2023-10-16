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

def esmfold_match(session, sequences, color_confidence=True, trim = True,
                  pae = False, ignore_cache=False):
    '''
    Find the most similar sequence in the ESMFold database to each specified
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

    # Sequence search ESM Metagenomic Atlas
    from .fetch import esmfold_fetch
    from .search import esmfold_sequence_search
    from chimerax.alphafold.match import _fetch_by_sequence
    seq_models = _fetch_by_sequence(session, sequences,
                                    color_confidence=color_confidence, trim=trim,
                                    sequence_search=esmfold_sequence_search,
                                    fetch=esmfold_fetch, ignore_cache=ignore_cache, log=log)

    # Report sequences with no ESMfold model
    seqs_no_model = [seq for seq in sequences if seq not in seq_models]
    if seqs_no_model:
        from chimerax.alphafold.match import _sequences_description
        msg = ('No ESM Metagenomic Atlas model with similar sequence for %s'
               % _sequences_description(seqs_no_model))
        log.warning(msg)

    from chimerax.alphafold.match import _group_chains_by_structure
    mlist, nmodels = _group_chains_by_structure(seq_models)
    session.models.add(mlist)

    from .search import _plural
    msg = 'Opened %d ESM Metagenomic Atlas model%s' % (nmodels, _plural(nmodels))
    log.info(msg)

    if pae:
        for seq,models in seq_models.items():
            for m in models:
                from .pae import esmfold_pae
                esmfold_pae(session, structure = m, mgnify_id = m.database.id,
                            version = m.database.version)

    return mlist

def register_esmfold_match_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg
    from chimerax.atomic import SequencesArg
    desc = CmdDesc(
        required = [('sequences', SequencesArg)],
        keyword = [('color_confidence', BoolArg),
                   ('trim', BoolArg),
                   ('pae', BoolArg),
                   ('ignore_cache', BoolArg)],
        synopsis = 'Fetch ESMFold database models matching an open structure'
    )
    register('esmfold match', desc, esmfold_match, logger=logger)
