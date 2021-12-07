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
# Search AlphaFold database for sequences using BLAST
#
def alphafold_search(session, sequence, cutoff=1.0e-3, max_sequences=100, matrix="BLOSUM62"):

    from chimerax.atomic import Chain
    if isinstance(sequence, Chain):
        chain_spec = sequence.string(style = 'command', include_structure = True)
    else:
        chain_spec = None
    seq_name = (getattr(sequence, 'uniprot_name', None)
                or getattr(sequence, 'uniprot_accession', None))
    from chimerax.blastprotein import BlastProteinJob
    BlastProteinJob(session, sequence.ungapped(), chain_spec, database='alphafold',
                    cutoff=cutoff, matrix=matrix, max_seqs=max_sequences,
                    sequence_name = seq_name)
    
# -----------------------------------------------------------------------------
#
def register_alphafold_search_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, IntArg, EnumOf
    from chimerax.atomic import SequenceArg
    from chimerax.blastprotein import AvailableMatrices
    desc = CmdDesc(
        required = [('sequence', SequenceArg)],
        keyword = [("cutoff", FloatArg),
                   ("matrix", EnumOf(AvailableMatrices)),
                   ("max_sequences", IntArg)],
        synopsis = 'Search AlphaFold database for a sequence using BLAST'
    )
    register('alphafold search', desc, alphafold_search, logger=logger)
