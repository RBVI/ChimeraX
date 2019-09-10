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

clustal_strong_groups = ["STA", "NEQK", "NHQK", "NDEQ", "QHRK", "MILV", "MILF", "HY", "FYW"]
clustal_weak_groups = ["CSA", "ATV", "SAG", "STNK", "STPA", "SGND", "SNDEQK", "NDEQHK",
    "NEQHRK", "FVLIM", "HFY"]

from chimerax.core.state import State
class Alignment(State):
    """A sequence alignment,
    
    Should only be created through new_alignment method of the alignment manager
    """

    def __init__(self, session, seqs, ident, file_attrs, file_markups, auto_destroy, auto_associate,
            description, intrinsic):
        self.session = session
        if isinstance(seqs, tuple):
            seqs = list(seqs)
        self._seqs = seqs
        self.ident = ident
        self.file_attrs = file_attrs
        self.file_markups = file_markups
        self.auto_destroy = auto_destroy
        self.description = description
        self.observers = []
        self.viewers = []
        self.viewers_by_subcommand = {}
        self.viewer_to_subcommand = {}
        self._observer_notification_suspended = 0
        self._ob_note_suspended_data = []
        self.associations = {}
        # need to be able to look up chain obj even after demotion to Sequence
        self._sseq_to_chain = {}
        from chimerax.atomic import Chain
        self.intrinsic = intrinsic
        self._in_destroy = False
        for i, seq in enumerate(seqs):
            if isinstance(seq, Chain):
                from copy import copy
                seqs[i] = copy(seq)
            seqs[i].match_maps = {}
        self._assoc_handler = None
        if auto_associate is None:
            # Create association for alignment's StructureSeqs, no auto-assoc
            self.associate(None, keep_intrinsic=True)
            self._auto_associate = False
        elif auto_associate:
            # if "session", don't associate with currently open structures (session restore
            # will do that), but allow future auto-association
            if auto_associate != "session":
                from chimerax.atomic import AtomicStructure
                self.associate([s for s in session.models if isinstance(s, AtomicStructure)], force=False)
            # get the auto-association working...
            self._auto_associate = False
            self.auto_associate = True
        else:
            self._auto_associate = False

    def associate(self, models, seq=None, force=True, min_length=10, reassoc=False,
            keep_intrinsic=False):
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

           Normally, associating additional chains would change the 'intrinsic' property to
           False, but some operations (like the inital associations) need to keep it True.
        """

        if not keep_intrinsic:
            self.intrinsic = False
        from chimerax.atomic import Sequence, Chain, StructureSeq, AtomicStructure, \
            SeqMatchMap, estimate_assoc_params, StructAssocError, try_assoc
        from .settings import settings
        status = self.session.logger.status
        reeval = False
        if isinstance(models, Chain):
            structures = [models]
        elif models is None:
            for seq in self._seqs:
                if isinstance(seq, StructureSeq) \
                and seq.existing_residues.chains[0] not in self.associations:
                    self.associate([], seq=seq, reassoc=reassoc, keep_intrinsic=keep_intrinsic)
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
                match_map = SeqMatchMap(seq, seq)
                for res, index in seq.res_map.items():
                    match_map.match(res, index)
                self.prematched_assoc_structure(match_map, 0, reassoc)
                new_match_maps.append(match_map)
            else:
                aseqs = [seq]
        else:
            aseqs = self.seqs
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
            if '.' in struct.id_string:
                # ensemble
                struct_name += " (" + struct.id_string + ")"
            def do_assoc(add_gaps=False):
                if reeval and sseq in self.associations:
                    old_aseq = self.associations[sseq]
                    if old_aseq == best_match_map.align_seq:
                        return
                    self.disassociate(sseq)
                if not self.intrinsic:
                    s1, s2, s3 = ("", "", "") if best_errors == 1 else ("es", "and/", "s")
                    if add_gaps:
                        gaps = " %sor gap%s" % (s2, s3)
                    else:
                        gaps = ""
                    status("Associated %s %s to %s with %d mismatch%s%s" % (struct_name,
                        sseq.name, best_match_map.align_seq.name, best_errors, s1, gaps), log=True)
                self.prematched_assoc_structure(best_match_map, best_errors, reassoc)
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
                best_errors = best_match_map = None
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
                        match_map, errors = try_assoc(aseq, sseq,
                            (est_len, segments, gaps), max_errors=try_errors)
                    except StructAssocError:
                        # maybe the sequence is derived from the structure...
                        if gaps:
                            try:
                                match_map, errors = try_assoc(aseq, sseq,
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
                                match_map, errors = try_assoc(aseq, sseq,
                                    (len(sseq), [sseq[:]], []), max_errors=errors-1)
                            except StructAssocError:
                                pass

                    best_match_map = match_map
                    best_errors = errors
                    if errors == 0:
                        break

                if best_match_map:
                    do_assoc()
            if not associated and force:
                # nothing matched built-in criteria, use Needleman-Wunsch
                best_errors = None
                max_errors = len(sseq) // settings.assoc_error_rate
                for sseq in sseqs:
                    # aseqs are already sorted by length...
                    for aseq in aseqs:
                        status("Using Needleman-Wunsch to test-associate"
                            " %s %s with %s\n" % (struct_name, sseq.name, aseq.name))
                        match_map, errors = nw_assoc(self.session, aseq, sseq)
                        if best_errors is None or errors < best_errors:
                            best_match_map = match_map
                            best_errors = errors
                if best_match_map:
                    do_assoc(add_gaps=True)
                else:
                    status("No reasonable association found for %s %s" % (struct_name, sseq.name))

        if new_match_maps:
            if reassoc:
                note_name = "modify association"
                note_data = ("add association", new_match_maps)
            else:
                note_name = "add association"
                note_data = new_match_maps
            self._notify_observers(note_name, note_data)

    def add_observer(self, observer):
        """Called by objects that care about alignment changes that are not themselves viewer
           (e.g. alignment headers).  Most of the documentation for attach_viewer() applies."""
        self.observers.append(observer)

    def attach_viewer(self, viewer, *, subcommand_name=None):
        """Called by the viewer (with the viewer instance as the arg) to receive notifications
           from the alignment (via the viewers alignment_notification method).  When the viewer
           is done with the alignment (typically at viewer exit), it should call the
           detach_viewer method (which may destroy the alignment if the alignment's
           auto_destroy attr is True).

           Also needs to be called by the viewer at session restore, since the alignment doesn't
           keep the registered viewers (to avoid a circular dependency).

           If the viewer handles commands, you need to provide the sequence subcommand used by
           your viewer (same as given to the manager's 'register_viewer' call) as 'subcommand_name'.
        """
        self.viewers.append(viewer)
        self.observers.append(viewer)
        if subcommand_name:
            self.viewers_by_subcommand.setdefault(subcommand_name, []).append(viewer)
            self.viewer_to_subcommand[viewer] = subcommand_name

    @property
    def auto_associate(self):
        return self._auto_associate

    @auto_associate.setter
    def auto_associate(self, assoc):
        if assoc == self._auto_associate:
            return
        if assoc:
            from chimerax.core.models import ADD_MODELS
            from chimerax.atomic import AtomicStructure
            self._assoc_handler = self.session.triggers.add_handler(ADD_MODELS, lambda tname, models:
                self.associate([s for s in models if isinstance(s, AtomicStructure)], force=False))
        else:
            self._assoc_handler.remove()
            self._assoc_handler = None
        self._auto_associate = assoc

    @property
    def being_destroyed(self):
        return self._in_destroy

    def detach_viewer(self, viewer):
        """Called when a viewer is done with the alignment (see attach_viewer)"""
        self.viewers.remove(viewer)
        self.observers.remove(viewer)
        sc = self.viewer_to_subcommand.get(viewer, None)
        if sc:
            self.viewers_by_subcommand[sc].remove(viewer)
        if not self.viewers and self.auto_destroy and not self._in_destroy:
            self.session.alignments.destroy_alignment(self)

    def disassociate(self, sseq, *, reassoc=False, demotion=False):
        if sseq not in self.associations or self._in_destroy:
            return

        if self.intrinsic:
            self.session.alignments.destroy_alignment(self)
            return
        aseq = self.associations[sseq]
        match_map = aseq.match_maps[sseq]
        del aseq.match_maps[sseq]
        del self.associations[sseq]
        match_map.mod_handler.remove()
        if reassoc:
            return
        if not demotion:
            self._notify_observers("remove association", [match_map])

            # if the structure seq hasn't been demoted/destroyed, log the disassociation
            struct = sseq.structure
            struct_name = struct.name
            if '.' in struct.id_string:
                # ensemble
                struct_name += " (" + struct.id_string + ")"
            self.session.logger.info("Disassociated %s %s from %s" % (struct_name, sseq.name, aseq.name))
        # delay notifying the observers until all chain demotions/deletions have been received
        def _delay_disassoc(_, __, match_map=match_map):
            self._notify_observers("remove association", [match_map])
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER
        from chimerax import atomic
        atomic.get_triggers().add_handler('changes', _delay_disassoc)

    def match(self, ref_chain, match_chains, *, iterate=-1, restriction=None):
        """Match the match_chains onto the ref_chain.  All chains must already be associated
           with the alignment.

           If 'iterate' is -1, then preference setting for iteration will be used.
           If 'iterate' is None, then no iteration occurs.  Otherwise, it is the cutoff value
           where iteration stops.

           'restriction', if provided, is a list of gapped column positions that the matching
           should be limited to.

           This returns a series of tuples, one per match chain, describing the resulting
           match.  The values in the 5-tuple are:

              * atoms used from the match chain
              * atoms used from the reference chain
              * RMSD across those atoms
              * RMSD across the original (possibly restricted) atoms, before iteration pruning
              * a :py:class:`~chimerax.core.Place` object describing the transformation to place
                the match chain onto the reference chain

            These values can all be None, if the matching failed (usually too few atoms to match).
        """
        if ref_chain not in self.associations:
            raise ValueError("%s not associated with any sequence" % ref_chain.full_name)

        match_structures = set([mc.structure for mc in match_chains])
        if len(match_structures) != len(match_chains):
            raise ValueError("Match chains must all come from different structures")
        if ref_chain.structure in match_structures:
            raise ValueError("Match chains and reference chain must come from different structures")

        if iterate == -1:
            from .settings import settings
            iterate = settings.iterate

        return_vals = []
        ref_seq = self.associations[ref_chain]
        if restriction is not None:
            ref_ungapped_positions = [ref_seq.gapped_to_ungapped(i) for i in restriction]
        for match_chain in match_chains:
            if match_chain not in self.associations:
                raise ValueError("%s not associated with any sequence" % match_chain.full_name)
            match_seq = self.associations[match_chain]
            if restriction is not None:
                match_ungapped_positions = [match_seq.gapped_to_ungapped(i) for i in restriction]
                restriction_set = set()
                for ur, um in zip(ref_ungapped_positions, match_ungapped_positions):
                    if ur is not None and um is not None:
                        restriction_set.add(ur)
            ref_atoms = []
            match_atoms = []
            ref_res_to_pos = ref_seq.match_maps[ref_chain].res_to_pos
            match_pos_to_res = match_seq.match_maps[match_chain].pos_to_res
            for rres, rpos in ref_res_to_pos.items():
                if restriction is not None and rpos not in restriction_set:
                    continue
                gpd = ref_seq.ungapped_to_gapped(rpos)
                mug = match_seq.gapped_to_ungapped(gpd)
                mpos = match_seq.gapped_to_ungapped(ref_seq.ungapped_to_gapped(rpos))
                if mpos is None:
                    continue
                mres = match_pos_to_res[mpos]
                if mres is None:
                    continue
                ref_atoms.append(rres.principal_atom)
                match_atoms.append(mres.principal_atom)
            from chimerax.core.commands import align
            from chimerax.atomic import Atoms
            try:
                return_vals.append(align.align(self.session, Atoms(match_atoms), Atoms(ref_atoms),
                    cutoff_distance=iterate))
            except align.IterationError:
                return_vals.append((None, None, None, None, None))
        return return_vals

    def prematched_assoc_structure(self, match_map, errors, reassoc):
        """If somehow you had obtained a SeqMatchMap for the align_seq<->struct_seq correspondence,
           you would use this call directly instead of the more usual associate() call
        """
        aseq = match_map.align_seq
        sseq = match_map.struct_seq
        chain = sseq.chain
        self._sseq_to_chain[sseq] = chain
        aseq.match_maps[chain] = match_map
        self.associations[chain] = aseq

        # set up callbacks for structure changes
        match_map.mod_handler = match_map.triggers.add_handler('modified', self._mmap_mod_cb)

    def remove_observer(self, observer):
        """Called when an observer is done with the alignment (see add_observer)"""
        self.observers.remove(observer)

    def resume_notify_observers(self):
        self._observer_notification_suspended -= 1
        if self._observer_notification_suspended == 0:
            cur_note = None
            for note, data, viewer_criteria in self._ob_note_suspended_data:
                if cur_note == None:
                    cur_note = note
                    cur_data = data
                    continue
                if note == cur_note:
                    if isinstance(cur_data, list):
                        cur_data.extend(data)
                        continue
                    if isinstance(cur_data, tuple) and len(cur_data) == 2\
                    and cur_data[0] == data[0] and isinstance(cur_data[1], list):
                        cur_data[1].extend(data[1])
                        continue
                self._notify_observers(cur_note, cur_data, viewer_criteria=viewer_criteria)
                cur_note = note
                cur_data = data
            if cur_note is not None:
                self._notify_observers(cur_note, cur_data, viewer_criteria=viewer_criteria)
            self._ob_note_suspended_data = []

    def save(self, path_or_stream, format_name="fasta"):
        import importlib
        mod = importlib.import_module(".io.save%s" % format_name.upper(), "chimerax.seqalign")
        from chimerax.core.io import open_filename
        stream = open_filename(path_or_stream, "w")
        with stream:
            mod.save(self.session, self, stream)

    @property
    def seqs(self):
        return self._seqs[:]

    def suspend_notify_observers(self):
        self._observer_notification_suspended += 1

    def _destroy(self):
        self._in_destroy = True
        self._notify_observers("destroyed", None)
        self.viewers = []
        self.observers = []
        aseqs = set()
        for sseq, aseq in self.associations.items():
            aseq.match_maps[sseq].mod_handler.remove()
            aseqs.add(aseq)
        for aseq in aseqs:
            aseq.match_maps.clear()
        self.associations.clear()
        if self._assoc_handler:
            self._assoc_handler.remove()

    def _dispatch_viewer_command(self, session, viewer_keyword, subcommand_text):
        from chimerax.core.errors import UserError
        viewers = self.viewers_by_subcommand.get(viewer_keyword, [])
        if not viewers:
            raise UserError("No '%s' viewers attached to alignment '%s'"
                % (viewer_keyword, self.ident))
        self._notify_observers("command", subcommand_text, viewer_criteria=viewer_keyword)

    def _mmap_mod_cb(self, trig_name, match_map):
        if len(match_map) == 0:
            self.disassociate(self._sseq_to_chain[match_map.struct_seq], demotion=True)
            del self._sseq_to_chain[match_map.struct_seq]
        else:
            self._notify_observers("modify association", ("modify association", [match_map]))

    def _notify_observers(self, note_name, note_data, *, viewer_criteria=None):
        if self._observer_notification_suspended > 0:
            self._ob_note_suspended_data.append((note_name, note_data, viewer_criteria))
            return
        if viewer_criteria is None:
            recipients = self.observers
        else:
            recipients = self.viewers_by_subcommand.get(viewer_criteria, [])
        for recipient in recipients:
            recipient.alignment_notification(note_name, note_data)
            if note_name in ["add association", "remove association"]:
                recipient.alignment_notification("modify association", (note_name, note_data))
            elif note_name in ["add sequences", "remove sequences"]:
                recipient.alignment_notification("add or remove sequences", (note_name, note_data))

    @staticmethod
    def restore_snapshot(session, data):
        """For restoring scenes/sessions"""
        ident = data['ident'] if 'ident' in data else data['name']
        aln = Alignment(session, data['seqs'], ident, data['file attrs'],
            data['file markups'], data['auto_destroy'],
            "session" if data['auto_associate'] else False,
            data.get('description', ident), data.get('intrinsic', False))
        aln.associations = data['associations']
        for s, mm in zip(aln.seqs, data['match maps']):
            s.match_maps = mm
            for chain, match_map in mm.items():
                match_map.mod_handler = match_map.triggers.add_handler('modified', aln._mmap_mod_cb)
        return aln

    def __str__(self):
        return self.ident

    def take_snapshot(self, session, flags):
        """For session/scene saving"""
        return { 'version': 1, 'seqs': self._seqs, 'ident': self.ident,
            'file attrs': self.file_attrs, 'file markups': self.file_markups,
            'associations': self.associations, 'match maps': [s.match_maps for s in self._seqs],
            'auto_destroy': self.auto_destroy, 'auto_associate': self.auto_associate,
            'description' : self.description, 'intrinsic' : self.intrinsic }


def nw_assoc(session, align_seq, struct_seq):
    '''Wrapper around Needleman-Wunsch matching, to make it return the same kinds of values
       that try_assoc returns'''

    from chimerax.atomic import Sequence, SeqMatchMap
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

    match_map = SeqMatchMap(align_seq, struct_seq)
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
