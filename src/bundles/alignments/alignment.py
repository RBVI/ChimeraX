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

from chimerax.core.state import State
class Alignment(State):
    """A sequence alignment,
    
    Should only be created through new_alignment method of the alignment manager
    """

    def __init__(self, session, seqs, name, file_attrs, file_markups, auto_destroy, auto_associate):
        self.session = session
        self.name = name
        self.file_attrs = file_attrs
        self.file_markups = file_markups
        self.auto_destroy = auto_destroy
        self.auto_associate = auto_associate
        self.viewers = []
        self.associations = {}
        from chimerax.core.atomic import Chain
        for i, seq in enumerate(seqs):
            if isinstance(seq, Chain):
                from copy import copy
                seqs[i] = copy(seq)
            seq.match_maps = {}
        if self.auto_associate is None:
            self.associate(None)
            self.auto_associate = False
        elif self.auto_associate:
            from chimerax.core.atomic import AtomicStructure
            self.associate([s for s in session.models
                if isinstance(s, AtomicStructure)], force=False)
            from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
            self.session.triggers.add_handler(ADD_MODELS, lambda tname, models:
                self.associate([s for s in models if isinstance(s, AtomicStructure)], force=False))

    def associate(self, models, seq=None, force=True, min_length=10, reassoc=False):
        """associate models with sequences

           'models' is normally a list of AtomicStructures, but it can be a Chain or None.
           If None, then some or all of the alignment sequences have 'residues'
           attributes indicating their corresponding model; set up the proper associations.

           if 'seq' is given, associate only with that sequence, otherwise consider all
           sequences in alignment.  If a non-empty list of models is provided and 'force'
           is False, then it is assumed that the seq has just been added to the alignment
           and that associations are being re-evaluated.

           If force is True, then if no association meets the built-in association criteria,
           then use Needleman-Wunsch to force an association with at least one sequence.

           If a chain is less than 'min_length' residues, ignore it.
        """

        from chimerax.core.atomic import Chain, StructureSeq, AtomicStructure, SeqMatchMap, \
            estimate_assoc_params, StructAssocError, try_assoc
        from .settings import settings
        status = self.session.logger.status
        reeval = False
        if isinstance(models, Chain):
            structures = [models]
        elif models is None:
            for seq in self.seqs:
                if isinstance(seq, StructureSeq) \
                and seq.existing_residues.chains[0] not in self.associations:
                    self.associate([], seq=seq, reassoc=reassoc)
            return
        else:
            structures = [m for m in models if isinstance(m, AtomicStructure)]
            if structures and seq and not force:
                reeval = True
        # sort alignment sequences from shortest to longest; a match against a shorter
        # sequence is better than a match against a longer sequence for the same number
        # of errors (except for structure sequences _longer_ than alignment sequences,
        # handled later)
        new_match_maps = []
        if seq:
            if isinstance(seq, StructureSeq) and not isinstance(models, Sequence):
                # if the sequence we're being asked to set up an association for is a
                # StructureSeq then we already know what structure it associates with and how...
                structures = []
                match_map = SeqMatchMap(self.session, seq, seq)
                for res, index in seq.res_map.items():
                    match_map.match(res, index)
                #TODO: finish prematched_...
                self.prematched_assoc_structure(seq, seq, match_map, 0, reassoc)
                new_match_maps.append(match_map)
            else:
                aseqs = [seq]
        else:
            aseqs = self.seqs[:]
            aseqs.sort(key=lambda s: len(s.ungapped()))
        if structures:
            forw_aseqs = aseqs
            rev_aseqs = aseqs[:]
            rev_aseqs.reverse()
        for struct in structures:
            if isinstance(struct, Chain):
                sseqs = [struct]
                struct = struct.structure
            else:
                sseqs = list(struct.chains)
                # sort sequences so that longest is tried first
                sseqs.sort(key=lambda s: len(s), reverse=True)
            associated = False
            struct_name = struct.name
            if '.' in struct.id_string():
                # ensemble
                struct_name += " (" + struct.id_string() + ")"
            def do_assoc():
                if reeval and sseq in self.associations:
                    old_aseq = self.associations[sseq]
                    if old_aseq == best_seq:
                        return
                    #TODO
                    #self.disassociate(sseq)
                msg = "Associated %s %s to %s with %d error(s)" % (struct_name, sseq.name,
                        best_seq.name, best_errors)
                status(msg, log=True, follow_with= "Right-click to focus on residue\n"
                    "Right-shift-click to focus on region", follow_log=False, blank_after=10)
                self.prematched_assoc_structure(best_seq, sseq,
                        best_match_map, best_errors, reassoc)
                new_match_maps.append(best_match_map)
                nonlocal associated
                associated = True
            for sseq in sseqs:
                if len(sseq) < min_length:
                    continue

                # find the apparent gaps in the structure, and estimate total length of
                # structure sequence given these gaps; make a list of the continuous segments
                est_len, segments, gaps = estimate_assoc_params(sseq)
                if not force:
                    if len(segments) > 10 and len(segments[0]) == 1 \
                    and segments.count(segments[0]) == len(segments):
                        # some kind of bogus structure (e.g. from SAXS)
                        return

                if est_len >= len(forw_aseqs[-1].ungapped()):
                    # structure sequence longer than alignment sequence;
                    # match against longest alignment sequences first
                    aseqs = rev_aseqs
                elif est_len > len(forw_aseqs[0].ungapped()):
                    # mixture of longer and shorter alignment seqs;
                    # do special sorting
                    mixed = aseqs[:]
                    def mixed_key_func(s):
                        ls = len(s)
                        uls = len(s.ungapped())
                        if uls >= est_len:
                            # larger than estimated length; want smallest (closest to est_len)
                            # first; and before all the ones smaller than est_len; so use
                            # negative numbers
                            return uls - ls
                        # smaller than est_len; want largest (closest to est_len) first
                        return ls - uls
                    mixed.sort(key=mixed_key_func)
                    aseqs = mixed
                else:
                    aseqs = forw_aseqs
                best_seq = best_errors = None
                max_errors = len(sseq) // settings.assoc_error_rate
                if reeval:
                    aseqs = []
                    for chain in self.associations.keys():
                        if chain.structure == struct:
                            aseqs.append(self.associations[chain])
                    aseqs.append(seq)
                for aseq in aseqs:
                    if best_errors:
                        try_errors = best_errors - 1
                    else:
                        try_errors = max_errors
                    try:
                        match_map, errors = try_assoc(self.session, aseq, sseq,
                            (est_len, segments, gaps), max_errors=try_errors)
                    except StructAssocError:
                        # maybe the sequence is derived from the structure...
                        if gaps:
                            try:
                                match_map, errors = try_assoc(self.session, aseq, sseq,
                                    (len(sseq), [sseq[:]], []), max_errors=try_errors)
                            except StructAssocError:
                                continue
                        else:
                            continue
                    else:
                        # if the above worked but had errors, see if just
                        # smooshing sequence together works better
                        if errors and gaps:
                            try:
                                match_map, errors = try_assoc(self.session, aseq, sseq,
                                    (len(sseq), [sseq[:]], []), max_errors=errors-1)
                            except StructAssocError:
                                pass

                    best_match_map = match_map
                    best_errors = errors
                    best_seq = aseq
                    if errors == 0:
                        break

                if best_seq:
                    do_assoc()
            if not associated and force:
                # nothing matched built-in criteria, use Needleman-Wunsch
                best_seq = best_sseq = best_errors = None
                max_errors = len(sseq) // settings.assoc_error_rate
                for sseq in sseqs:
                    # aseqs are already sorted by length...
                    for aseq in aseqs:
                        status("Using Needleman-Wunsch to test-associate"
                            " %s %s with %s\n" % (struct_name, sseq.name, aseq.name))
                        match_map, errors = nw_assoc(self.session, aseq, sseq)
                        if not best_seq or errors < best_errors:
                            best_match_map = match_map
                            best_errors = errors
                            best_seq = aseq
                            best_sseq = sseq
                if best_match_map:
                    do_assoc()
                else:
                    status("No reasonable association found for %s %s" % (struct_name, sseq.name))

        if new_match_maps:
            if reassoc:
                note_name = "mod assoc"
                note_data = ("add assoc", new_match_maps)
            else:
                note_name = "add assoc"
                note_data = new_match_maps
            self._notify_viewers(note_name, note_data)

    def attach_viewer(self, viewer):
        """Called by the viewer (with the viewer instance as the arg) to receive notifications
           from the alignment (via the viewers alignment_notification method).  When the viewer
           is done with the alignment (typically at viewer exit), it should call the
           detach_viewer method (which may destroy the alignment if the alignment's
           auto_destroy attr is True).
        """
        self.viewers.append(viewer)

    def detach_viewer(self, viewer):
        """Called when a viewer is done with the alignment (see attach_viewer)"""
        self.viewers.remove(viewer)
        if not self.viewers and self.auto_destroy:
            self.session.alignments.destroy_alignment(self)

    def prematched_assoc_structure(self, aseq, sseq, match_map, errors, reassoc):
        """If somehow you had obtained a SeqMatchMap for the aseq<->sseq correspondence,
           you would use this call instead of the more usual associate() call
        """
        chain = sseq.chain
        aseq.match_maps[chain] = match_map
        self.associations[chain] = aseq

        # set up callbacks for structure changes
        """
        match_map["mavDelHandler"] = mseq.notegers.addHandler(
                mseq.TRIG_DELETE, self._mseqDelCB, match_map)
        match_map["mavModHandler"] = mseq.triggers.addHandler(
                mseq.TRIG_MODIFY, self._mseqModCB, match_map)
        """

    def take_snapshot(self, session, flags):
        return { 'version': 1, 'seqs': self.seqs, 'name': self.name,
            'file attrs': self.file_atts, 'file markups': self.file_markups,
            'associations': self.associations, 'match maps': [s.match_maps for s in self.seqs]}

    def reset_state(self, session):
        pass

    @staticmethod
    def restore_snapshot(session, data):
        aln = Alignment(data['seqs'], data['name'], data['file attrs'], data['file markups'])
        aln.associations = associations
        for s, mm in zip(aln.seqs, data['match maps']):
            s.match_maps = mm
        return aln

    def _destroy(self):
        for viewer in self.viewers[:]:
            viewer.alignment_notification(self, "destroyed", None)
        self.viewers = []

    def _notify_viewers(self, note_name, note_data):
        for viewer in self.viewers:
            viewer.alignment_notification(note_name, note_data)
            if note_name in ["add assoc", "del assoc"]:
                viewer.alignment_notification("mod assoc", (note_name, note_data))

def nw_assoc(session, align_seq, struct_seq):
    '''Wrapper around Needle-Wunch matching, to make it return the same kinds of values
       that try_assoc returns'''

    from chimerax.core.atomic import Sequence, SeqMatchMap
    sseq = struct_seq
    aseq = Sequence(name=align_seq.name, characters=align_seq.ungapped())
    aseq.circular = align_seq.circular
    from chimerax.seqalign.align_algs.NeedlemanWunsch import nw
    score, match_list = nw(sseq, aseq)

    errors = 0
    # matched are in reverse order...
    try:
        m_end = match_list[0][0]
    except IndexError:
        m_end = -1
    if m_end < len(sseq) - 1:
        # trailing unmatched
        errors += len(sseq) - m_end - 1

    match_map = SeqMatchMap(session, aseq, sseq)
    last_match = m_end + 1
    for s_index, a_index in match_list:
        if sseq[s_index] != aseq[a_index]:
            errors += 1

        if s_index < last_match - 1:
            # gap in structure sequence
            errors += last_match - s_index - 1

        res = sseq.residues[s_index]
        if res:
            match_map.match(res, a_index)

        last_match = s_index
    if last_match > 0:
        # beginning unmatched
        errors += last_match

    if len(sseq) > len(aseq):
        # unmatched residues forced, reduce errors by that amount...
        errors -= len(sseq) - len(aseq)

    return match_map, errors
