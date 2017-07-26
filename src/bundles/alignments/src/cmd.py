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

    if len(chains) == 1:
        chain = chains[0]
        ident = ".".join([str(part) for part in chain.structure.id]) + "." + chain.chain_id
        alignment = session.alignments.new_alignment([chain], ident, seq_viewer="mav",
            auto_associate=None, intrinsic=True)
    else:
        # all chains have to have the same sequence, and they will all be associated with
        # that sequence
        sequences = set([chain.characters for chain in chains])
        if len(sequences) != 1:
            from chimerax.core.errors import UserError
            raise UserError("Chains must have same sequence")
        chars = sequences.pop()
        chain_ids = set([chain.chain_id for chain in chains])
        if len(chain_ids) < len(chains) or len(chain_ids) > 10:
            name = "%d chains" % len(chains)
        else:
            name = "chains %s" % ",".join(sorted(list(chain_ids)))
        from chimerax.core.atomic import Sequence
        seq = Sequence(name=name, characters=chars)
        def get_numbering_start(chain):
            for i, r in enumerate(chain.residues):
                if r is None or r.deleted:
                    continue
                return r.number - i
            return None
        starts = set([get_numbering_start(chain) for chain in chains])
        starts.discard(None)
        if len(starts) == 1:
            seq.numbering_start = starts.pop()
        alignment = session.alignments.new_alignment([seq], None, seq_viewer="mav",
            auto_associate=False, name=chains[0].description, intrinsic=True)
        alignment.suspend_notify_viewers()
        for chain in chains:
            alignment.associate(chain, keep_intrinsic=True)
        alignment.resume_notify_viewers()

def register_seqalign_command(logger):
    from chimerax.core.commands import CmdDesc, register, UniqueChainsArg
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        synopsis = 'show structure chain sequence'
    )
    register('sequence chain', desc, seqalign_chain, logger=logger)
