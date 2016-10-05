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
    def __init__(self, session, seqs, name, file_attrs, file_markups, autodestroy):
        self.session = session
        self.seqs = seqs
        self.name = name
        self.file_attrs = file_attrs
        self.file_markups = file_markups

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
            estimate_assoc_params, StructAssocError
        from .settings import settings
        status = self.session.logger.status
        reeval = False
        if isinstance(models, Chain):
            structures = [models]
        elif models is None:
            for seq in self.seqs:
                if isinstance(seq, Chain) \
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
            aseqs.sort(lambda a, b: cmp(len(a.ungapped()), len(b.ungapped())))
        if structures:
            forw_aseqs = aseqs
            rev_aseqs = aseqs[:]
            rev_aseqs.reverse()
        for struct in structures:
            if isinstance(struct, Chain):
                sseqs = [struct]
                struct = struct.structure
            else:
                sseqs = struct.chains
                # sort sequences so that longest is tried first
                sseqs.sort(lambda a, b: 0 - cmp(len(a),len(b)))
            assoc_info = None
            struct_name = os.path.split(struct.name)[-1]
            if '.' in struct.id_string():
                # ensemble
                struct_name += " (" + struct.id_string() + ")"
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
                    def _mix_sort(a, b, lm):
                        la = len(a.ungapped())
                        lb = len(b.ungapped())
                        if la >= lm:
                            if lb >= lm:
                                return cmp(la, lb)
                            else:
                                return -1
                        else:
                            if lb >= lm:
                                return 1
                            else:
                                return cmp(lb, la)
                    mixed = aseqs[:]
                    mixed.sort(lambda a, b: _mix_sort(a, b, est_len))
                    aseqs = mixed
                else:
                    aseqs = forw_aseqs
                best_seq = best_errors = None
                max_errors = len(sseq) // settings.assoc_error_rate
                if reeval:
                    aseqs = []
                    for chain in self.associations.keys():
                        if chain.structure == struct:
                            aseqs.append(self.associations[chain]
                    aseqs.append(seq)
                for aseq in aseqs:
                    if best_errors:
                        try_errors = best_errors - 1
                    else:
                        try_errors = max_errors
                    try:
                        match_map, errors = try_assoc(aseq, sseq, segments, gaps, est_len,
                            max_errors=try_errors)
                    except StructAssocError:
                        # maybe the sequence is derived from the structure...
                        if gaps:
                            try:
                                match_map, errors = try_assoc(aseq, sseq, [sseq[:]], [], len(sseq),
                                    max_errors=try_errors)
                            except StructAssocError:
                                continue
                        else:
                            continue
                    else:
                        # if the above worked but had errors, see if just
                        # smooshing sequence together works better
                        if errors and gaps:
                            try:
                                match_map, errors = try_assoc(aseq, sseq, [sseq[:]], [], len(sseq),
                                    max_errors=errors-1)
                            except StructAssocError:
                                pass

                    best_match_map = match_map
                    best_errors = errors
                    best_seq = aseq
                    if errors == 0:
                        break

                if best_seq:
                    if assoc_info and best_errors >= assoc_info[-1]:
                        continue
                    assoc_info = (best_seq, sseq, best_match_map, best_errors)
            if not assoc_info and force:
                # nothing matched built-in criteria, use Needleman-Wunsch
                best_seq = best_sseq = best_errors = None
                max_errors = len(sseq) // settings.assoc_error_rate
                for sseq in sseqs:
                    # aseqs are already sorted by length...
                    for aseq in aseqs:
                        status("Using Needleman-Wunsch to test-associate"
                            " %s %s with %s\n" % (struct_name, sseq.name, aseq.name))
                        #TODO
                        match_map, errors = nwAssoc(aseq, sseq)
                        if not best_seq \
                        or errors < best_errors:
                            best_match_map = match_map
                            best_errors = errors
                            best_seq = aseq
                            best_sseq = sseq
                if best_match_map:
                    assoc_info = (best_seq, best_sseq, best_match_map, best_errors)
                else:
                    status("No reasonable association"
                        " found for %s %s\n" % (struct_name, sseq.name))

            if assoc_info:
                best_seq, sseq, best_match_map, best_errors = assoc_info
                if reeval and sseq.molecule in self.associations:
                    old_aseq = self.associations[sseq.molecule]
                    if old_aseq == best_seq:
                        continue
                    self.disassociate(sseq.molecule)
                msg = "Associated %s %s to %s with %d error(s)"\
                        "\n" % (struct_name, sseq.name,
                        best_seq.name, best_errors)
                status(msg, log=1, followWith=
                    "Right-click to focus on residue\n"
                    "Right-shift-click to focus on region",
                    followLog=False, blankAfter=10)
                self.prematched_assoc_structure(best_seq, sseq,
                        best_match_map, best_errors, reassoc)
                new_match_maps.append(best_match_map)
        if self.intrinsicStructure and len(self.seqs) == 1:
            self.showSS()
            status("Helices/strands depicted in gold/green")
        if new_match_maps:
            if reassoc:
                trigName = MOD_ASSOC
                trigData = (ADD_ASSOC, new_match_maps)
            else:
                trigName = ADD_ASSOC
                trigData = new_match_maps
            self.triggers.activateTrigger(trigName, trigData)
            if self.prefs[SHOW_SEL]:
                self.regionBrowser.showChimeraSelection()

    def prematched_assoc_structure(self, aseq, sseq, match_map, errors, reassoc):
        """If somehow you had obtained a SeqMatchMap for the aseq<->sseq correspondence,
           you would use this call instead of the more usual associate() call
        """
        chain = sseq.chain
        try:
            aseq.match_maps[chain] = match_map
        except AttributeError:
            aseq.match_maps = { chain: match_map }
        self.associations[chain] = aseq

        #TODO
        #self.seqCanvas.assocSeq(aseq)
        #TODO
        # set up callbacks for structure changes
        """
        match_map["mavDelHandler"] = mseq.triggers.addHandler(
                mseq.TRIG_DELETE, self._mseqDelCB, match_map)
        match_map["mavModHandler"] = mseq.triggers.addHandler(
                mseq.TRIG_MODIFY, self._mseqModCB, match_map)
        """

    def take_snapshot(self, session, flags):
        return { 'version': 1, 'seqs': self.seqs, 'name': self.name,
            'file_attrs': self.file_atts, 'file_markups': self.file_markups }

    def reset_state(self, session):
        pass

    @staticmethod
    def restore_snapshot(session, data):
        return Alignment(data['seqs'], data['name'], data['file_attrs'], data['file_markups'])

    def _close(self):
        """Called by alignments manager so alignment can clean up (notify viewers, etc.)"""
        pass
