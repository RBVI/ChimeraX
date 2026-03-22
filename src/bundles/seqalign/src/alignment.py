# vim: set expandtab shiftwidth=4 softtabstop=4:

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

clustal_strong_groups = ["STA", "NEQK", "NHQK", "NDEQ", "QHRK", "MILV", "MILF", "HY", "FYW"]
clustal_weak_groups = ["CSA", "ATV", "SAG", "STNK", "STPA", "SGND", "SNDEQK", "NDEQHK",
    "NEQHRK", "FVLIM", "HFY"]

from chimerax.core.errors import NotABug, UserError
from chimerax.core.state import State
class Alignment(State):
    """A sequence alignment,
    
    Should only be created through new_alignment method of the alignment manager
    """

    # non-viewers (e.g. headers) notified before viewers, so that they can be "ready" for the viewer
    # (important for NOTE_REALIGNMENT)
    NOTE_ADD_ASSOC     = "add association"
    NOTE_DEL_ASSOC     = "remove association"
    NOTE_MOD_ASSOC     = "modify association"
    NOTE_ADD_SEQS      = "add seqs"
    NOTE_PRE_DEL_SEQS  = "pre-remove seqs"
    NOTE_DEL_SEQS      = "remove seqs"
    NOTE_ADD_DEL_SEQS  = "add or remove seqs"
    NOTE_EDIT_START    = "editing started"
    NOTE_EDIT_END      = "editing finished"
    NOTE_DESTROYED     = "destroyed"
    NOTE_COMMAND       = "command"
    NOTE_REF_SEQ       = "reference seq changed"
    NOTE_SEQ_CONTENTS  = "seq contents changed"  # Not fired if NOTE_REALIGNMENT applicable
    NOTE_SEQ_NAME      = "sequence name changed"
    NOTE_REALIGNMENT   = "sequences realigned"  # preempts NOTE_SEQ_CONTENTS
    NOTE_RMSD_UPDATE   = "rmsd change"  # RMSD value changed, or chains relevant to RMSD may have changed

    # associated note_data for the above is None except for:
    #   NOTE_ADD_ASSOC: list of new matchmaps
    #   NOTE_DEL_ASSOC: a dictionary with information about an association being deleted.  Because of the
    #           way structure deletion works, one of the values of the dictionary is an estimate.  The
    #           key/values are:
    #       'match map': the deleted association's matchmap
    #       'num remaining associations': number of chains still associated immediately after the deletion
    #       'max previous structures': estimate of the number of associated _structures_ just before deletion
    #       'num remaining structures': number of associated _structures just after deletion
    #   NOTE_MOD_ASSOC: a 2-tuple of the the specific type of modification and associated data.  The type
    #       can be NOTE_ADD_ASSOC or NOTE_DEL_ASSOC, in which case the second value of the tuple is the
    #       data normally provided with that notification, or NOTE_MOD_ASSOC, for which the second value
    #       is a list of modified matchmaps.
    #   NOTE_COMMAND: the observer subcommand text
    #   NOTE_REF_SEQ: the new reference sequence (which could be None)
    #   NOTE_SEQ_CONTENTS: the sequence whose characters changed
    #   NOTE_SEQ_NAME: the sequence whose name changed
    #   NOTE_REALIGNMENT: a list of copies of the previous sequences
    #   not yet implemented:  NOTE_ADD_SEQS, NOTE_PRE_DEL_SEQS, NOTE_DEL_SEQS, NOTE_ADD_DEL_SEQS,

    NOTE_HDR_VALUES    = "header values changed"
    NOTE_HDR_NAME      = "header name changed"
    NOTE_HDR_RELEVANCE = "header's relevance to alignment changed"
    NOTE_HDR_SHOWN     = "header's 'shown' attribute changed"

    # associated note_data for the above is just the header except for:
    #   NOTE_HDR_VALUES: a 2-tuple of the header and where the values changed (either a 2-tuple of 0-based
    #       indices into the header, or None -- which indicates the entire header)

    class NoSelectionError(NotABug): pass
    class NoSelectionExpansionError(NotABug): pass

    COL_IDENTITY_ATTR = "seq_identity"

    def __init__(self, session, seqs, ident, file_attrs, file_markups, auto_destroy, auto_associate,
            description, intrinsic, *, create_headers=True, session_restore=False):
        if not seqs:
            raise ValueError("Cannot create alignment of zero sequences")
        self.session = session
        self._session_restore = session_restore
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
        self._column_counts_cache = None
        self._observer_notification_suspended = 0
        self._ob_note_suspended_data = []
        self._modified_mmaps = []
        self._mmap_handler = None
        self._reference_seq = None
        self._rmsd_chains = None
        self._rmsd_handler = None
        self.associations = {}
        # need to be able to look up chain obj even after demotion to Sequence
        self._sseq_to_chain = {}
        from chimerax.atomic import Chain, StructureSeq
        self.intrinsic = intrinsic
        self._in_destroy = False
        self._seq_handlers = []
        self.match_maps = {}
        for i, seq in enumerate(self._seqs):
            self.match_maps[seq] = {}
            self._seq_handlers.append(
                self._seqs[i].triggers.add_handler("rename", self._seq_name_changed_cb))
            if isinstance(self._seqs[i], StructureSeq):
                self._seq_handlers.append(self._seqs[i].triggers.add_handler("characters changed",
                    self._seq_characters_changed_cb))
        # need an _headers placeholder before associate() gets called...
        self._headers = []
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
                structures = [s for s in session.models if isinstance(s, AtomicStructure)]
                if structures:
                    self.associate(structures, force=False)
            # get the auto-association working...
            self._auto_associate = False
            self.auto_associate = True
        else:
            self._auto_associate = False
        if create_headers:
            self._headers = [hdr_class(self) for hdr_class in session.alignments.headers()]
            if file_markups is not None:
                from chimerax.core.attributes import string_to_attr
                for name, markup in file_markups.items():
                    self._headers.append(MarkupHeaderSequence(self, name, markup,
                        identifier=string_to_attr(name, prefix="file_markup_")))
            self._headers.sort(key=lambda hdr: hdr.name.casefold())
            attr_headers = []
            for header in self._headers:
                header.shown = header.settings.initially_shown and header.relevant
            if len(seqs) > 1:
                from chimerax.atomic import Residue
                Residue.register_attr(self.session, self.COL_IDENTITY_ATTR, "sequence alignment",
                    attr_type=float, can_return_none=False)
            if not session_restore:
                self._set_residue_attributes()

    def add_fixed_header(self, name, contents, *, shown=True, identifier=None, hdr_class=None):
        if len(contents) != len(self._seqs[0]):
            raise ValueError(f"Fixed header '{name}' is not the same length as alignment")
        if hdr_class is None:
            from chimerax.alignment_headers import FixedHeaderSequence as hdr_class
        header = hdr_class(self, name, contents, identifier)
        self._headers.append(header)
        header.shown = shown
        return header

    def add_headers_menu_entry(self, menu):
        headers_menu = menu.addMenu("Headers")
        headers = self.headers
        headers.sort(key=lambda hdr: hdr.ident.casefold())
        from chimerax.core.commands import run, StringArg
        align_arg = "%s " % self if len(self.session.alignments.alignments) > 1 else ""
        from Qt.QtGui import QAction
        for hdr in headers:
            action = QAction(hdr.name, headers_menu)
            action.setCheckable(True)
            action.setChecked(hdr.shown)
            if not hdr.relevant:
                action.setEnabled(False)
            action.triggered.connect(lambda *, action=action, hdr=hdr, align_arg=align_arg, ses=self.session:
                run(ses, "seq header %s%s %s" % (align_arg, hdr.ident,
                "show" if action.isChecked() else "hide")))
            headers_menu.addAction(action)
        headers_menu.addSeparator()
        hdr_save_menu = headers_menu.addMenu("Save")
        for hdr in headers:
            if not hdr.relevant:
                continue
            action = QAction(hdr.name, hdr_save_menu)
            action.triggered.connect(lambda *, hdr=hdr, align_arg=align_arg, ses=self.session: run(
                ses, "seq header %s%s save browse" % (align_arg, hdr.ident)))
            hdr_save_menu.addAction(action)
        return headers_menu

    def add_observer(self, observer):
        """Called by objects that care about alignment changes that are not themselves viewer
           (e.g. alignment headers).  Most of the documentation for attach_viewer() applies."""
        self.observers.append(observer)

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
            if isinstance(seq, StructureSeq) and not models:
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
                note_name = self.NOTE_MOD_ASSOC
                note_data = (self.NOTE_ADD_ASSOC, new_match_maps)
            else:
                note_name = self.NOTE_ADD_ASSOC
                note_data = new_match_maps
            # since StructureSeq demotion notifications may be delayed until the 'changes done'
            # trigger, we need to do a hacky check here and possibly also delay this notification
            for seq in self.associations:
                if getattr(seq, 'structure', None) is None:
                    # it's been demoted
                    def _delay_assoc(_, __, *, name=note_name, data=note_data):
                        self._notify_observers(name, data)
                        from chimerax.core.triggerset import DEREGISTER
                        return DEREGISTER
                    from chimerax import atomic
                    atomic.get_triggers().add_handler('changes done', _delay_assoc)
                    break
            else:
                self._notify_observers(note_name, note_data)

    def associated_residues(self, aseqs=None):
        if aseqs is None:
            aseqs = self.seqs
        residues = []
        for aseq in aseqs:
            for match_map in self.match_maps[aseq].values():
                residues.extend(match_map.res_to_pos.keys())
        return residues

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

    def column_counts(self):
        """Returns a dictionary keyed on column index, with values that are 2-tuples of numpy
           arrays; the first member is the array of unique characters in that column, and the
           second array is the corresponding counts for those characters.
        """
        if self._column_counts_cache is None:
            import numpy
            data = numpy.array([list(seq.characters) for seq in self._seqs])
            cache = self._column_counts_cache = {}
            for i in range(len(data[0])):
                cache[i] = numpy.unique(data[:,i], return_counts=True)
        return self._column_counts_cache.copy()

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

        aseq = self.associations[sseq]
        if self.intrinsic and len(self.match_maps[aseq]) == 1:
            if demotion:
                self.session.alignments.destroy_alignment(self)
                return
            self.intrinsic = False
        match_map = self.match_maps[aseq][sseq]
        del self.match_maps[aseq][sseq]
        del self.associations[sseq]
        match_map.mod_handler.remove()
        if reassoc:
            return
        if not demotion:
            # if the structure seq hasn't been demoted/destroyed, log the disassociation
            struct = sseq.structure
            struct_name = struct.name
            if '.' in struct.id_string:
                # ensemble
                struct_name += " (" + struct.id_string + ")"
            self.session.logger.info("Disassociated %s %s from %s" % (struct_name, sseq.name, aseq.name))
        num_unknown = 0
        structures = set()
        for sseq in self.associations:
            structure = getattr(sseq, 'structure', None)
            if structure is None:
                # demoted
                num_unknown += 1
            else:
                structures.add(structure)
        data = {
            'match map': match_map,
            'num remaining associations': len(self.associations),
            'max previous structures': len(structures) + num_unknown,
            'num remaining structures': len(structures)
        }
        if not demotion:
            # do immediately, since 'changes done' trigger may never fire for manual disassociations
            self._notify_observers(self.NOTE_DEL_ASSOC, data)
            return
        # delay notifying the observers until all chain demotions/deletions have been received
        def _delay_disassoc(_, __, data=data):
            self._notify_observers(self.NOTE_DEL_ASSOC, data)
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER
        from chimerax import atomic
        atomic.get_triggers().add_handler('changes done', _delay_disassoc)

    def expand_selection_by_columns(self):
        from chimerax.atomic import selected_residues, Residues
        sel_residues = set(selected_residues(self.session))
        if not sel_residues:
            raise self.NoSelectionError("No selection to expand")

        sel_columns = set()
        for aseq in self.seqs:
            for mm in self.match_maps[aseq].values():
                for sr in sel_residues:
                    try:
                        ungapped = mm[sr]
                    except KeyError:
                        continue
                    sel_columns.add(aseq.ungapped_to_gapped(ungapped))

        expansion = []
        for aseq in self.seqs:
            for mm in self.match_maps[aseq].values():
                for sc in sel_columns:
                    ungapped = aseq.gapped_to_ungapped(sc)
                    if ungapped is None:
                        continue
                    try:
                        sr = mm[ungapped]
                    except KeyError:
                        continue
                    if sr not in sel_residues:
                        expansion.append(sr)
        if not expansion:
            raise self.NoSelectionExpansionError("No extra residues selected by expanding columns")
        from chimerax.std_commands.select import select_add
        from chimerax.core.objects import Objects
        select_add(self.session, Objects(atoms=Residues(expansion).atoms))

    @property
    def headers(self):
        return self._headers[:]

    def match(self, ref_chain, match_chains, *, iterate=-1, conservation=None, restriction=None):
        """Match the match_chains onto the ref_chain.  All chains must already be associated
           with the alignment.

           If 'iterate' is -1, then preference setting for iteration will be used.
           If 'iterate' is None, then no iteration occurs.  Otherwise, it is the cutoff value
           where iteration stops.

           'conservation', if provided, is the percent identity that a column has to have to
           be included in the matching.

           'restriction', if provided, is a list of gapped column positions that the matching
           should be (further) limited to.

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
            raise UserError("%s not associated with any sequence" % ref_chain.full_name)

        match_structures = set([mc.structure for mc in match_chains])
        if len(match_structures) != len(match_chains):
            raise UserError("Match chains must all come from different structures")
        if ref_chain.structure in match_structures:
            raise UserError("Match chains and reference chain must come from different structures")

        if iterate == -1:
            from .settings import settings
            iterate = settings.iterate

        if restriction is None:
            if conservation is None:
                final_restriction = None
            else:
                threshold = len(self._seqs) * conservation
                final_restriction = set([col for col in range(len(self._seqs[0]))
                    if self.most_common(col)[-1] >= threshold])
        else:
            if conservation is None:
                final_restriction = set(restriction)
            else:
                threshold = len(self._seqs) * conservation
                final_restriction = set([col for col in restriction
                    if self.most_common(col)[-1] >= threshold])

        return_vals = []
        ref_seq = self.associations[ref_chain]
        if final_restriction is not None:
            ref_ungapped_positions = [ref_seq.gapped_to_ungapped(i) for i in final_restriction]
        for match_chain in match_chains:
            if match_chain not in self.associations:
                raise UserError("%s not associated with any sequence" % match_chain.full_name)
            match_seq = self.associations[match_chain]
            if final_restriction is not None:
                match_ungapped_positions = [match_seq.gapped_to_ungapped(i) for i in final_restriction]
                restriction_set = set()
                for ur, um in zip(ref_ungapped_positions, match_ungapped_positions):
                    if ur is not None and um is not None:
                        restriction_set.add(ur)
            ref_atoms = []
            match_atoms = []
            ref_res_to_pos = self.match_maps[ref_seq][ref_chain].res_to_pos
            match_pos_to_res = self.match_maps[match_seq][match_chain].pos_to_res
            for rres, rpos in ref_res_to_pos.items():
                if final_restriction is not None and rpos not in restriction_set:
                    continue
                gpd = ref_seq.ungapped_to_gapped(rpos)
                mug = match_seq.gapped_to_ungapped(gpd)
                mpos = match_seq.gapped_to_ungapped(ref_seq.ungapped_to_gapped(rpos))
                if mpos is None:
                    continue
                try:
                    mres = match_pos_to_res[mpos]
                except KeyError:
                    continue
                ref_atoms.append(rres.principal_atom)
                match_atoms.append(mres.principal_atom)
            from chimerax.std_commands import align
            from chimerax.atomic import Atoms
            try:
                return_vals.append(align.align(self.session, Atoms(match_atoms), Atoms(ref_atoms),
                    cutoff_distance=iterate))
            except align.IterationError:
                return_vals.append((None, None, None, None, None))
        return return_vals

    def most_common(self, col_index, *, non_gap=True):
        """Returns most common character in the column given by 'col_index' and the count for that character.
           For ties, one of the most common characters, chosen arbitrarily, will be returned.  If 'non_gap'
           is True, gap characters will be ignored.
        """
        chars, counts = self.column_counts()[col_index]
        if non_gap:
            max_count = 0
            for char, count in zip(chars, counts):
                if not char.isalpha():
                    continue
                if count > max_count:
                    max_count = count
                    max_char = char
            if max_count == 0:
                return ' ', 0
            return max_char, max_count
        import numpy
        max_index = numpy.argmax(counts)
        return chars[max_index], counts[max_index]

    def notify(self, note_name, note_data):
        """Used by headers to issue notifications, but theoretically could be used by anyone"""
        self._notify_observers(note_name, note_data)
        if note_name == self.NOTE_HDR_RELEVANCE:
            hdr = note_data
            if hdr.shown and not hdr.relevant:
                hdr.shown = False
        elif note_name == self.NOTE_HDR_SHOWN:
            hdr = note_data
            if not hdr.eval_while_hidden:
                self._set_residue_attributes(headers=[hdr])
            if not self._session_restore:
                if hdr.shown:
                    msg = 'Showing %s header ("%s" residue attribute) for alignment %s' % (hdr.ident,
                        hdr.residue_attr_name, self)
                else:
                    msg = 'Hiding %s header for alignment %s' % (hdr.ident, self)
                self.session.logger.info(msg)
        elif note_name == self.NOTE_HDR_VALUES:
            hdr, bounds = note_data
            self._set_residue_attributes(headers=[hdr])

    def prematched_assoc_structure(self, match_map, errors, reassoc):
        """If somehow you had obtained a SeqMatchMap for the align_seq<->struct_seq correspondence,
           you would use this call directly instead of the more usual associate() call
        """
        aseq = match_map.align_seq
        sseq = match_map.struct_seq
        chain = sseq.chain
        self._sseq_to_chain[sseq] = chain
        self.match_maps[aseq][chain] = match_map
        self.associations[chain] = aseq

        # set up callbacks for structure changes
        match_map.mod_handler = match_map.triggers.add_handler('modified', self._mmap_mod_cb)
        self._set_residue_attributes(match_maps=[match_map])

    @property
    def reference_seq(self):
        return self._reference_seq

    @reference_seq.setter
    def reference_seq(self, ref_seq):
        # can be None
        if ref_seq == self._reference_seq:
            return
        self._reference_seq = ref_seq
        self._notify_observers(self.NOTE_REF_SEQ, ref_seq)

    def remove_observer(self, observer):
        """Called when an observer is done with the alignment (see add_observer)"""
        # Observers may be slow to remove themselves if the removal is because
        # of garbage collection [#18702], so this test...
        if not self._in_destroy:
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

    @property
    def rmsd_chains(self):
        prev_rmsd_chains = self._rmsd_chains
        if self._rmsd_chains is None:
            by_struct = {}
            for chain in self.associations:
                by_struct.setdefault(chain.structure, []).append(chain)
            chain_lists = list(by_struct.values())
            if len(chain_lists) < 2:
                self._rmsd_chains = []
            else:
                chain_lists.sort(key=lambda x: len(x))
                cl1, cl2 = chain_lists[:2]
                lowest = None
                for c1 in cl1:
                    for c2 in cl2:
                        rmsd = self._eval_rmsd([c1, c2])
                        if rmsd is None:
                            continue
                        if lowest is None or rmsd < lowest:
                            lowest = rmsd
                            best_chains = [c1, c2]
                if lowest is None:
                    best_chains = [cl1[0], cl2[0]]
                for cl in chain_lists[2:]:
                    lowest = None
                    for c in cl:
                        rmsd = self._eval_rmsd(best_chains + [c])
                        if rmsd is None:
                            continue
                        if lowest is None or rmsd < lowest:
                            lowest = rmsd
                            best_chain = c
                    if lowest is None:
                        best_chains.append(cl[0])
                    else:
                        best_chains.append(best_chain)
                self._rmsd_chains = best_chains
        if prev_rmsd_chains:
            if not self._rmsd_chains:
                if self._rmsd_handler is not None:
                    self._rmsd_handler.remove()
                    self._rmsd_handler = None
        elif self._rmsd_chains:
            from chimerax.atomic import get_triggers
            if self._rmsd_handler is None:
                self._rmsd_handler = get_triggers().add_handler('changes', self._rmsd_atomic_cb)
        return self._rmsd_chains

    def save(self, output, format_name="fasta"):
        import importlib
        mod = importlib.import_module(".io.save%s" % format_name.upper(),
            "chimerax.seqalign")
        from chimerax import io
        with io.open_output(output, 'utf-8') as stream:
            mod.save(self.session, self, stream)

    @property
    def seqs(self):
        return self._seqs[:]

    def suspend_notify_observers(self):
        self._observer_notification_suspended += 1

    def _atomic_changes_done(self, *args):
        self._notify_observers(self.NOTE_MOD_ASSOC, (self.NOTE_MOD_ASSOC, self._modified_mmaps))
        self._modified_mmaps = []
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _destroy(self):
        self._in_destroy = True
        self._notify_observers(self.NOTE_DESTROYED, None)
        self.viewers = []
        self.observers = []
        for header in self._headers:
            header.destroy()
        aseqs = set()
        for sseq, aseq in self.associations.items():
            self.match_maps[aseq][sseq].mod_handler.remove()
            aseqs.add(aseq)
        for aseq in aseqs:
            self.match_maps[aseq].clear()
        self.associations.clear()
        for handler in [self._assoc_handler, self._mmap_handler, self._rmsd_handler]:
            if handler:
                handler.remove()
        for handler in self._seq_handlers:
            handler.remove()

    def _dispatch_header_command(self, subcommand_text):
        from chimerax.core.commands import EnumOf
        enum = EnumOf(self.headers, ids=[header.ident for header in self._headers])
        header, ident_text, remainder = enum.parse(subcommand_text, self.session)
        header.process_command(remainder)

    def _dispatch_viewer_command(self, viewer_keyword, subcommand_text):
        viewers = self.viewers_by_subcommand.get(viewer_keyword, [])
        if not viewers:
            raise UserError("No '%s' viewers attached to alignment '%s'"
                % (viewer_keyword, self.ident))
        self._notify_observers(self.NOTE_COMMAND, subcommand_text, viewer_criteria=viewer_keyword)

    def _eval_rmsd(self, chains):
        chain_types = set([c.polymer_type for c in chains])
        if len(chain_types) > 1:
            return None
        ct = chain_types.pop()
        from chimerax.atomic import Residue
        if ct == Residue.PT_NONE:
            return None
        pa_name = { Residue.PT_AMINO: "CA", Residue.PT_NUCLEIC: "C4'" }[ct]
        total = 0.0
        n = 0
        from chimerax.geometry import distance_squared
        for coords in self._gather_coords(chains, pa_name):
            for i, crd1 in enumerate(coords):
                for crd2 in coords[i+1:]:
                    total += distance_squared(crd1, crd2)
            n += (len(coords) * (len(coords)-1)) // 2
        if n == 0:
            return None
        from math import sqrt
        return sqrt(total / n)

    def _gather_coords(self, chains, pa_name):
        coord_lists = []
        seqs = [self.associations[chain] for chain in chains]
        match_maps = [self.match_maps[self.associations[chain]][chain] for chain in chains]
        for pos in range(len(self.seqs[0])):
            crd_list = []
            for seq, mmap in zip(seqs, match_maps):
                ungapped = seq.gapped_to_ungapped(pos)
                if ungapped is None:
                    continue
                try:
                    r = mmap[ungapped]
                except KeyError:
                    continue
                if r:
                    pa = r.find_atom(pa_name)
                    if pa:
                        crd_list.append(pa.scene_coord)
            if len(crd_list) > 1:
                coord_lists.append(crd_list)
        return coord_lists

    def _mmap_mod_cb(self, trig_name, match_map):
        if len(match_map) == 0:
            self.disassociate(self._sseq_to_chain[match_map.struct_seq], demotion=True)
            del self._sseq_to_chain[match_map.struct_seq]
        else:
            if not self._modified_mmaps:
                if self._mmap_handler is None:
                    from chimerax.atomic import get_triggers
                    self._mmap_handler = get_triggers().add_handler(
                        "changes done", self._atomic_changes_done)
            self._modified_mmaps.append(match_map)

    def _notify_observers(self, note_name, note_data, *, viewer_criteria=None):
        if self._observer_notification_suspended > 0:
            self._ob_note_suspended_data.append((note_name, note_data, viewer_criteria))
            return
        if viewer_criteria is None:
            recipients = self.observers
            recipients.sort(key=lambda x: x in self.viewers)
        else:
            recipients = self.viewers_by_subcommand.get(viewer_criteria, [])
        for recipient in recipients:
            recipient.alignment_notification(note_name, note_data)
            if note_name in [self.NOTE_ADD_ASSOC, self.NOTE_DEL_ASSOC]:
                recipient.alignment_notification(self.NOTE_MOD_ASSOC, (note_name, note_data))
                self._notify_rmsd_change()
            elif note_name in [self.NOTE_ADD_SEQS, self.NOTE_DEL_SEQS]:
                recipient.alignment_notification(self.NOTE_ADD_DEL_SEQS, (note_name, note_data))

    def _notify_rmsd_change(self):
        if self._rmsd_chains is None:
            # no one currently interested in RMSD
            return
        self._rmsd_chains = None # force recomputation of relevant chains
        self._notify_observers(self.NOTE_RMSD_UPDATE, None)

    @staticmethod
    def restore_snapshot(session, data):
        """For restoring scenes/sessions"""
        ident = data['ident'] if 'ident' in data else data['name']
        create_headers = data['version'] < 2
        aln = Alignment(session, data['seqs'], ident, data['file attrs'],
            data['file markups'], data['auto_destroy'],
            "session" if data['auto_associate'] else False,
            data.get('description', ident), data.get('intrinsic', False), create_headers=create_headers,
            session_restore=True)
        aln.associations = data['associations']
        for s, mm in zip(aln.seqs, data['match maps']):
            aln.match_maps[s] = mm
            for chain, match_map in mm.items():
                match_map.mod_handler = match_map.triggers.add_handler('modified', aln._mmap_mod_cb)
        if 'sseq to chain' in data:
            aln._sseq_to_chain = data['sseq to chain']
        from chimerax.core.toolshed import get_toolshed
        ts = get_toolshed()
        if not create_headers:
            for bundle_name, class_name, header_state in data['headers']:
                bundle = ts.find_bundle(bundle_name, session.logger, installed=True)
                if not bundle:
                    bundle = ts.find_bundle(bundle_name, session.logger, installed=False)
                    if bundle:
                        session.logger.error("You need to install bundle %s in order to restore"
                            " alignment header of type %s" % (bundle_name, class_name))
                    else:
                        session.logger.error("Cannot restore alignment header of type %s due to"
                            " being unable to find any bundle named %s" % (class_name, bundle_name))
                    continue
                header_class = bundle.get_class(class_name, session.logger)
                if header_class:
                    aln._headers.append(header_class.session_restore(session, aln, header_state))
                else:
                    session.logger.warning("Could not find alignment header class %s in bundle %s"
                        % (class_name, bundle_name))
        aln._session_restore = False
        return aln

    def _rmsd_atomic_cb(self, trig_name, changes):
        if not self._rmsd_chains:
            # not currently relevant to anyone
            return
        if 'scene_coord changed' not in changes.structure_reasons():
            return
        from chimerax.atomic import StructureSeq
        for chain in self.associations:
            if chain.deleted or getattr(chain, 'structure', None) is None:
                # the ensuing disassociation/demotion will update the RMSD
                return
        for chain in self.associations:
            if chain.structure in changes.modified_structures():
                self._notify_rmsd_change()
                break

    def _seq_characters_changed_cb(self, trig_name, seq):
        self._column_counts_cache = None
        if not getattr(self, '_realigning', False):
            self._notify_observers(self.NOTE_SEQ_CONTENTS, seq)

    def _seq_name_changed_cb(self, trig_name, seq):
        self._notify_observers(self.NOTE_SEQ_NAME, seq)

    def _set_realigned(self, realigned_seqs):
        # realigned sequences need to be in the same order as the current sequences
        self._realigning = True
        from copy import copy
        prev_seqs = []
        for cur_seq, realigned_seq in zip(self.seqs, realigned_seqs):
            prev_seqs.append(copy(cur_seq))
            cur_seq.characters = realigned_seq.characters
        self._realigning = False
        self._notify_observers(self.NOTE_REALIGNMENT, prev_seqs)

    def _set_residue_attributes(self, *, headers=None, match_maps=None):
        if match_maps is None:
            match_maps = [mm for aseq in self.associations.values() for mm in self.match_maps[aseq].values()]
        if not match_maps:
            return
        def process_attr(attr_name, col_vals):
            assigned = set()
            for match_map in match_maps:
                aseq = match_map.align_seq
                for i, val in enumerate(col_vals):
                    ui = aseq.gapped_to_ungapped(i)
                    if ui is None:
                        continue
                    try:
                        r = match_map[ui]
                    except KeyError:
                        continue
                    if not hasattr(r, attr_name) or getattr(r, attr_name) != val:
                        setattr(r, attr_name, val)
                        assigned.add(r)
            if assigned:
                self.session.change_tracker.add_modified(assigned, attr_name + " changed")
        if headers is None:
            headers = [hdr for hdr in self._headers if hdr.shown or hdr.eval_while_hidden]
            if len(self.seqs) > 1:
                num_seqs = len(self._seqs)
                values = [100.0 * self.most_common(col)[1] / num_seqs for col in range(len(self._seqs[0]))]
                process_attr(self.COL_IDENTITY_ATTR, values)
        from chimerax.atomic import Residue
        for header in headers:
            attr_name = header.residue_attr_name
            Residue.register_attr(self.session, attr_name, "sequence alignment",
                attr_type=header.value_type, can_return_none=header.value_none_okay)
            process_attr(attr_name, header)

    def __str__(self):
        return self.ident

    def take_snapshot(self, session, flags):
        """For session saving"""
        from chimerax.core.toolshed import get_toolshed
        ts = get_toolshed()
        return { 'version': 2, 'seqs': self._seqs, 'ident': self.ident,
            'file attrs': self.file_attrs, 'file markups': self.file_markups,
            'associations': self.associations, 'match maps': [self.match_maps[s] for s in self._seqs],
            'auto_destroy': self.auto_destroy, 'auto_associate': self.auto_associate,
            'description' : self.description, 'intrinsic' : self.intrinsic,
            'sseq to chain': self._sseq_to_chain,
            'headers': [(ts.find_bundle_for_class(hdr.__class__).name,
                hdr.__class__.__name__, hdr.get_state()) for hdr in self.headers]
            }


def nw_assoc(session, align_seq, struct_seq):
    '''Wrapper around Needleman-Wunsch matching, to make it return the same kinds of values
       that try_assoc returns'''

    from chimerax.atomic import Sequence, SeqMatchMap
    sseq = struct_seq
    aseq = Sequence(name=align_seq.name, characters=align_seq.ungapped())
    aseq.circular = align_seq.circular
    from chimerax.alignment_algs.NeedlemanWunsch import nw
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

from chimerax.alignment_headers import FixedHeaderSequence
class MarkupHeaderSequence(FixedHeaderSequence):
    def settings_info(self):
        base_settings_name, defaults = super().settings_info()
        from chimerax.core.commands import BoolArg
        defaults.update({'initially_shown': (BoolArg, True)})
        return "sequence file header %s" % self.name, defaults
