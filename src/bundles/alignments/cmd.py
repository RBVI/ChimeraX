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

#
def seqalign_chain(session, chains):
    '''
    Show chain sequence(s)

    Parameters
    ----------
    chains : list of Chain
        Chains to show
    '''

    for chain in chains:
        ident = ".".join([str(part) for part in chain.structure.id]) + "." + chain.chain_id
        session.alignments.new_alignment([chain], ident, seq_viewer="mav", auto_associate=None)

def register_seqalign_command(logger):
    from chimerax.core.commands import CmdDesc, register, UniqueChainsArg
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        synopsis = 'show structure chain sequence'
    )
    register('seqalign chain', desc, seqalign_chain, logger=logger)
