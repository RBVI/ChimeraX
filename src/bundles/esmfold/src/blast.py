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
# Search ESMFold database for sequences using BLAST
#
def esmfold_search(session, sequence, cutoff=1.0e-3, max_sequences=100,
                   matrix="BLOSUM62", version=None):

    from chimerax.atomic import Chain
    if isinstance(sequence, Chain):
        chain_spec = sequence.string(style = 'command', include_structure = True)
    else:
        chain_spec = None
    seq_name = (getattr(sequence, 'uniprot_name', None)
                or getattr(sequence, 'uniprot_accession', None))
    if version is None:
        from .database import default_database_version
        version = default_database_version(session)
    from chimerax.blastprotein import BlastProteinJob
    BlastProteinJob(session, sequence.ungapped(), chain_spec, database='esmfold',
                    version=version, cutoff=cutoff, matrix=matrix, max_seqs=max_sequences,
                    sequence_name = seq_name)
    
# -----------------------------------------------------------------------------
#
def register_esmfold_search_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, IntArg, EnumOf, StringArg
    from chimerax.atomic import SequenceArg
    from chimerax.blastprotein import AvailableMatrices
    desc = CmdDesc(
        required = [('sequence', SequenceArg)],
        keyword = [("cutoff", FloatArg),
                   ("matrix", EnumOf(AvailableMatrices)),
                   ("max_sequences", IntArg),
                   ("version", StringArg)],
        synopsis = 'Search ESMFold database for a sequence using BLAST'
    )
    register('esmfold search', desc, esmfold_search, logger=logger)
