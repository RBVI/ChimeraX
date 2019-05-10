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

from chimerax.core.commands import Annotation, AnnotationError
class SeqArg(Annotation):
    '''A single sequence (in a single alignment)

       If only one alignment is open, the alignment ID can be omitted.
       Within the alignment, sequences can be specified by name or number, with
       negative numbers counting backwards from the end of the alignment.
    '''

    name = "[alignment-id]:sequence-name-or-number"
    _html_name = "[<i>alignment-id</i>]:<i>sequence-name-or-number</i>"

    @staticmethod
    def parse(text, session):
        align_seq, text, rest = AlignSeqPairArg.parse(text, session)
        return align_seq[-1], text, rest

class AlignSeqPairArg(Annotation):
    '''Same as SeqArg, but the return value is (alignment, seq)'''

    name = "[alignment-id:]sequence-name-or-number"
    _html_name = "[<i>alignment-id</i>:]<i>sequence-name-or-number</i>"

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import AnnotationError, next_token
        if not text:
            raise AnnotationError("Expected %s" % SeqArg.name)
        token, text, rest = next_token(text)
        if ':' not in token:
            align_id, seq_id = "", token
        else:
            align_id, seq_id = token.split(':', 1)
        if not align_id:
            aln_seq = None
            for aln in session.alignments.alignments:
                try:
                    seq = get_alignment_sequence(aln, seq_id)
                except MissingSequence:
                    pass
                else:
                    if aln_seq is None:
                        aln_seq = (aln, seq)
                    else:
                        raise AnnotationError("Multiple sequences match '%s'; please also specify the"
                            " alignment by prepending 'alignment-ID:'" % token)
            if aln_seq:
                return aln_seq, text, rest
            raise AnnotationError("No sequences match '%s'" % token)
        alignment = get_alignment_by_id(session, align_id)
        seq = get_alignment_sequence(alignment, seq_id)
        return (alignment, seq), text, rest

class AlignmentArg(Annotation):
    '''A sequence alignment'''

    name = "alignment-id"
    _html_name = "<i>alignment-id</i>"

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import AnnotationError, next_token
        if not text:
            raise AnnotationError("Expected %s" % SeqArg.name)
        token, text, rest = next_token(text)
        alignment = get_alignment_by_id(session, token)
        return alignment, text, rest

class MissingSequence(AnnotationError):
    pass
def get_alignment_sequence(alignment, seq_id):
    try:
        sn = int(seq_id)
    except ValueError:
        for seq in alignment.seqs:
            if seq.name == seq_id:
                break
        else:
            raise MissingSequence("No sequence named '%s' found in alignment" % seq_id)
    else:
        if sn == 0:
            raise AnnotationError("Sequence index must be positive or negative integer,"
                " not zero")
        if abs(sn) > len(alignment.seqs):
            raise AnnotationError("Sequence index (%d) larger than number of sequences"
                " in alignment (%d)" % (sn, len(alignment.seqs)))
        if sn > 0:
            seq = alignment.seqs[sn-1]
        else:
            seq = alignment.seqs[sn]
    return seq

def get_alignment_by_id(session, align_id, *, multiple_okay=False):
    if not align_id:
        if not session.alignments.alignments:
            raise AnnotationError("No alignments open!")
        elif len(session.alignments.alignments) > 1:
            if multiple_okay:
                return list(session.alignments.alignments)
            raise AnnotationError("More than one sequence alignment open;"
                " need to specify an alignment ID")
        alignment = session.alignments.alignments[0]
    else:
        try:
            alignment = session.alignments.alignments_map[align_id]
        except KeyError:
            raise AnnotationError("No known alignment with ID: '%s'" % align_id)
    if multiple_okay:
        return [alignment]
    return alignment

from chimerax.core.errors import UserError

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
        alignment = session.alignments.new_alignment([chain], ident, seq_viewer="sv",
            auto_associate=None, intrinsic=True)
    else:
        # all chains have to have the same sequence, and they will all be associated with
        # that sequence
        sequences = set([chain.characters for chain in chains])
        if len(sequences) != 1:
            raise UserError("Chains must have same sequence")
        chars = sequences.pop()
        chain_ids = set([chain.chain_id for chain in chains])
        if len(chain_ids) < len(chains) or len(chain_ids) > 10:
            name = "%d chains" % len(chains)
        else:
            name = "chains %s" % ",".join(sorted(list(chain_ids)))
        from chimerax.atomic import Sequence
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
        alignment = session.alignments.new_alignment([seq], None, seq_viewer="sv",
            auto_associate=False, name=chains[0].description, intrinsic=True)
        alignment.suspend_notify_observers()
        for chain in chains:
            alignment.associate(chain, keep_intrinsic=True)
        alignment.resume_notify_observers()

def seqalign_associate(session, chains, align_seq):
    aln, seq = align_seq
    for chain in chains:
        if chain in aln.associations:
            old_seq = aln.associations[chain]
            if old_seq == seq:
                session.logger.warning("%s already associated with %s" % (chain, seq.name))
                continue
            aln.disassociate(chain)
            session.logger.warning("Disassociated %s from %s" % (chain, old_seq))
        aln.associate(chain, seq=seq)

def seqalign_disassociate(session, chains, alignment=None):
    if alignment is None:
        alignments = session.alignments.alignments
    else:
        alignments = [alignment]

    for chain in chains:
        did_disassoc = False
        for aln in alignments:
            if chain in aln.associations:
                did_disassoc = True
                aln.disassociate(chain)
        if not did_disassoc:
            session.logger.warning("%s not associated with %s"
                % (chain, " any alignment" if alignment is None else "alignment %s" % alignment.ident))

def register_seqalign_command(logger):
    # REMINDER: update manager._builtin_subcommands as additional subcommands are added
    from chimerax.core.commands import CmdDesc, register
    from chimerax.atomic import UniqueChainsArg
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        synopsis = 'show structure chain sequence'
    )
    register('sequence chain', desc, seqalign_chain, logger=logger)
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg), ('align_seq', AlignSeqPairArg)],
        synopsis = 'associate chain(s) with sequence'
    )
    register('sequence associate', desc, seqalign_associate, logger=logger)
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        optional = [('alignment', AlignmentArg)],
        synopsis = 'disassociate chain(s) from alignment'
    )
    register('sequence disassociate', desc, seqalign_disassociate, logger=logger)

    from . import manager
    manager._register_viewer_subcommands(logger)
