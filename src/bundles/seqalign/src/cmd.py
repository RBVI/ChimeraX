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

from chimerax.core.commands import Annotation, AnnotationError, next_token, DynamicEnum
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

class SeqRegionArg(Annotation):
    '''Part(s) of a single sequence (in a single alignment)

       If only one alignment is open, the alignment ID can be omitted.
       Within the alignment, sequences can be specified by name or number, with
       negative numbers counting backwards from the end of the alignment.

       The region is a comma-separated list of sequence positions or ranges. If blank, the whole sequence.
       If the region text matches any of the text values in the special_region_values class variable
       (truncation allowed), the full value text is returned instead of the region list.


       Return value is (alignment, seq, list of (start,end) zero-based indices into sequence).
    '''

    name = "[alignment-id]:sequence-name-or-number:[sequence-positions-or-ranges]"
    _html_name = "[<i>alignment-id</i>]:<i>sequence-name-or-number</i>:<i>sequence-positions-or-ranges</i>"

    special_region_values = []

    @classmethod
    def parse(cls, text, session):
        if not text:
            raise AnnotationError("Expected %s" % cls.name)
        token, text, rest = next_token(text)
        try:
            align_seq_text, region_text = token.rsplit(':', 1)
        except ValueError:
            raise AnnotationError("Must include at least one ':' character")
        align_seq, _text, _rest = AlignSeqPairArg.parse(align_seq_text, session, empty_okay=True)
        if _rest:
            raise AnnotationError("Unexpected text (%s) after alignment/sequence name and before range"
                % _rest)
        align, seq = align_seq
        if not region_text:
            # whole sequence
            regions = [(0, len(seq))]
        else:
            for special in cls.special_region_values:
                if special.lower().startswith(region_text.lower()):
                    regions = special
                    break
            else:
                regions = []
                for segment in region_text.split(','):
                    if '-' in segment:
                        start, end = segment.split('-', 1)
                    else:
                        start = end = segment
                    try:
                        start, end = int(start)-1, int(end)-1
                    except ValueError:
                        raise AnnotationError("Sequence position is not a comma-separated list of integer"
                            " positions or position ranges")
                    if start < 0:
                        raise AnnotationError("Sequence position is less than one")
                    elif end >= len(seq):
                        raise AnnotationError("Sequence position (%d) is past end of sequence" % end+1)
                    elif end < start:
                        raise AnnotationError("End sequence position (%d) is less than start position"
                            % (end+1, start+1))
                    regions.append((start, end))
        return (align, seq, regions), text, rest

class AlignSeqPairArg(Annotation):
    '''Same as SeqArg, but the return value is (alignment, seq)'''

    name = "[alignment-id:]sequence-name-or-number"
    _html_name = "[<i>alignment-id</i>:]<i>sequence-name-or-number</i>"

    @staticmethod
    def parse(text, session, empty_okay=False):
        from chimerax.core.commands import AnnotationError, next_token
        if not text:
            if empty_okay:
                text = ':'
            else:
                raise AnnotationError("Expected %s" % AlignSeqPairArg.name)
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
            token = rest = ""
        else:
            token, text, rest = next_token(text)
        alignment = get_alignment_by_id(session, token)
        return alignment, text, rest

class BaseViewerArg(DynamicEnum):
    def __init__(self, session, viewer_type):
        def viewer_names(mgr, viewer_type):
            names = set()
            for vname, nicknames in mgr.viewer_info[viewer_type].items():
                names.add(vname)
                names.update(nicknames)
            return names
        super().__init__(lambda mgr=session.alignments, f=viewer_names, t=viewer_type: f(mgr, t))

class AlignmentViewerArg(BaseViewerArg):
    def __init__(self, session):
        super().__init__(session, "alignment")

class SequenceViewerArg(BaseViewerArg):
    def __init__(self, session):
        super().__init__(session, "sequence")

class MissingSequence(AnnotationError):
    pass
def get_alignment_sequence(alignment, seq_id):
    if not seq_id:
        if len(alignment.seqs) == 1:
            return alignment.seqs[0]
        raise MissingSequence("Sequence specifier omitted")
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

def seqalign_chain(session, chains, *, viewer=True):
    '''
    Show chain sequence(s)

    Parameters
    ----------
    chains : list of Chain
        Chains to show
    '''

    if len(chains) == 1:
        chain = chains[0]
        ident = ".".join([str(part) for part in chain.structure.id]) + "/" + chain.chain_id
        alignment = session.alignments.new_alignment([chain], ident, auto_associate=None, intrinsic=True,
            viewer=viewer)
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
        starts = set([chain.numbering_start for chain in chains])
        starts.discard(None)
        if len(starts) == 1:
            seq.numbering_start = starts.pop()
        alignment = session.alignments.new_alignment([seq], None, auto_associate=False,
            name=chains[0].description, intrinsic=True, viewer=viewer)
        alignment.suspend_notify_observers()
        for chain in chains:
            alignment.associate(chain, keep_intrinsic=True)
        alignment.resume_notify_observers()

def seqalign_associate(session, chains, target=None):
    if target is None:
        alignments = session.alignments.alignments
        seq = "best"
    elif type(target) == tuple:
        aln, seq = target
        alignments = [aln]
    else:
        seq = "best"
        alignments = [target]
    for aln in alignments:
        for chain in chains:
            if chain in aln.associations:
                old_seq = aln.associations[chain]
                aln.disassociate(chain)
            else:
                old_seq = None
            if seq == "best":
                aln.associate(chain)
            else:
                aln.associate(chain, seq=seq, reassoc=(seq==old_seq))

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

def seqalign_header(session, alignments, subcommand_text):
    from .alignment import Alignment
    if alignments is None:
        from .cmd import get_alignment_by_id
        alignments = get_alignment_by_id(session, "", multiple_okay=True)
        if not alignments:
            raise UserError("No alignments open")
    elif isinstance(alignments, Alignment):
        alignments = [alignments]
    for alignment in alignments:
        alignment._dispatch_header_command(subcommand_text)

from chimerax.atomic.seq_support import IdentityDenominator, percent_identity
def seqalign_identity(session, src1, src2=None, *, denominator=IdentityDenominator.default):
    "Either src1 is an alignment and src2 is None (report all vs. all), or src1 and src2 are sequences"
    from .alignment import Alignment
    usage = "Must either provide an alignment, an alignment and a sequence, or two sequence arguments"
    if src2 is None:
        if not isinstance(src1, Alignment):
            raise UserError(usage)
        for i, seq1 in enumerate(src1.seqs):
            for seq2 in src1.seqs[i+1:]:
                identity = percent_identity(seq1, seq2, denominator=denominator)
                session.logger.info("%s vs. %s: %.2f%% identity" % (seq1.name, seq2.name, identity))
        return
    if isinstance(src1, Alignment):
        seqs1 = src1.seqs
    else:
        seqs1 = [src1]
    for seq1 in seqs1:
        try:
            identity = percent_identity(seq1, src2, denominator=denominator)
        except ValueError as e:
            raise UserError(str(e))
        session.logger.info("%s vs. %s: %.2f%% identity" % (seq1.name, src2.name, identity))
    return identity

def seqalign_refseq(session, ref_seq_info):
    if isinstance(ref_seq_info, tuple):
        aln, ref_seq = ref_seq_info
    else:
        aln, ref_seq = ref_seq_info, None
    aln.reference_seq = ref_seq

MUSCLE = "MUSCLE"
CLUSTAL_OMEGA = "Clustal Omega"
alignment_program_name_args = { 'muscle': MUSCLE, 'omega': CLUSTAL_OMEGA, 'clustal': CLUSTAL_OMEGA,
    'clustalOmega': CLUSTAL_OMEGA }
def seqalign_align(session, seq_source, *, program=CLUSTAL_OMEGA, replace=False):
    from .alignment import Alignment
    if isinstance(seq_source, Alignment):
        raw_input_sequences = seq_source.seqs
        title = "%s realignment of %s" % (program, seq_source.description)
    else:
        raw_input_sequences = seq_source
        title = "%s alignment" % program
    from chimerax.atomic import Residue
    input_sequences = [s for s in raw_input_sequences
        if getattr(s, 'polymer_type', Residue.PT_PROTEIN) == Residue.PT_PROTEIN]
    if len(input_sequences) < 2:
        raise UserError("Must specify 2 or more protein sequences")
    from .align import realign_sequences
    realigned = realign_sequences(session, input_sequences, program=program)
    if replace:
        seq_source._set_realigned(realigned)
        return seq_source
    return session.alignments.new_alignment(realigned, None, name=title)

def register_seqalign_command(logger):
    # REMINDER: update manager._builtin_subcommands as additional subcommands are added
    from chimerax.core.commands import CmdDesc, register, create_alias, Or, EmptyArg, RestOfLine, ListOf, \
        EnumOf, BoolArg
    from chimerax.atomic import UniqueChainsArg, SequencesArg

    apns = list(alignment_program_name_args.keys())
    desc = CmdDesc(
        required = [('seq_source', Or(AlignmentArg, SequencesArg))],
        keyword = [('program', EnumOf([alignment_program_name_args[apn] for apn in apns], ids=apns)),
            ('replace', BoolArg)],
        synopsis = "align sequences"
    )
    register('sequence align', desc, seqalign_align, logger=logger)

    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        optional = [('target', Or(AlignmentArg, AlignSeqPairArg))],
        synopsis = 'associate chain(s) with sequence'
    )
    register('sequence associate', desc, seqalign_associate, logger=logger)

    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        keyword = [('viewer', Or(BoolArg, SequenceViewerArg(logger.session)))],
        synopsis = 'show structure chain sequence'
    )
    register('sequence chain', desc, seqalign_chain, logger=logger)

    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        optional = [('alignment', AlignmentArg)],
        synopsis = 'disassociate chain(s) from alignment'
    )
    register('sequence disassociate', desc, seqalign_disassociate, logger=logger)
    create_alias('sequence dissociate', 'sequence disassociate $*', logger=logger,
            url="help:user/commands/sequence.html#disassociate")

    desc = CmdDesc(
        required = [('alignments', Or(AlignmentArg,ListOf(AlignmentArg),EmptyArg)),
            ('subcommand_text', RestOfLine)],
        synopsis = "send subcommand to header"
    )
    register('sequence header', desc, seqalign_header, logger=logger)

    enum_members = [denom for denom in IdentityDenominator]
    desc = CmdDesc(
        required = [('src1', Or(AlignmentArg, SeqArg))],
        optional = [('src2', SeqArg)],
        keyword = [('denominator', EnumOf(enum_members, ids=[mem.value for mem in enum_members]))],
        synopsis = "report percent identity"
    )
    register('sequence identity', desc, seqalign_identity, logger=logger)

    desc = CmdDesc(
        required = [('ref_seq_info', Or(AlignSeqPairArg, AlignmentArg))],
        synopsis = "set alignment reference sequence"
    )
    register('sequence refseq', desc, seqalign_refseq, logger=logger)

    from . import manager
    manager._register_viewer_subcommands(logger)
