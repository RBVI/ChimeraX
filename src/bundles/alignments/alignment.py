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
        self.seqs = seqs
        self.name = name
        self.file_attrs = file_attrs
        self.file_markups = file_markups

    def associate(self, models, seq=None, force=True, min_length=10, reassoc=False):
        """associate models with sequences

           'models' is normally a list of models, but it can be a Chain or None.
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
        """
        from structAssoc import tryAssoc, nwAssoc, estimateAssocParams
        """
        from chimerax.core.atomic import Chain, StructureSeq, AtomicStructure
        """
        from prefs import ASSOC_ERROR_RATE
        """
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
            if isinstance(seq, StructureSeq) and not isinstance(models, Chain):
                # if the sequence we're being asked to set up an association for is a
                # StructureSeq then we already know what molecule it associates with and how...
                structures = []
                match_map = {}
                for res, index in seq.res_map.items():
                    match_map[res] = index
                    match_map[index] = res
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
        for mol in structures:
            if isinstance(mol, Sequence):
                mseqs = [mol]
                mol = mol.molecule
            else:
                mseqs = mol.sequences()
                # sort sequences so that longest is tried
                # first
                mseqs.sort(lambda a, b: 0 - cmp(len(a),len(b)))
            assocInfo = None
            molName = os.path.split(mol.name)[-1]
            if '.' in mol.oslIdent():
                # ensemble
                molName += " (" + mol.oslIdent() + ")"
            for mseq in mseqs:
                if len(mseq) < min_length:
                    continue

                # find the apparent gaps in the structure,
                # and estimate total length of structure
                # sequence given these gaps;
                # make a list of the continuous segments
                estLen, segments, gaps = estimateAssocParams(mseq)
                if not force:
                    if len(segments) > 10 and len(segments[0]) == 1 \
                    and segments.count(segments[0]) == len(segments):
                        # some kind of bogus structure (e.g. from SAXS)
                        return

                if estLen >= len(forw_aseqs[-1].ungapped()):
                    # structure sequence longer than
                    # alignment sequence; match against
                    # longest alignment sequences first
                    aseqs = rev_aseqs
                elif estLen > len(forw_aseqs[0].ungapped()):
                    # mixture of longer and shorter
                    # alignment seqs; do special sorting
                    mixed = aseqs[:]
                    mixed.sort(lambda a, b:
                            _mixSort(a, b, estLen))
                    aseqs = mixed
                else:
                    aseqs = forw_aseqs
                bestSeq = bestErrors = None
                maxErrors = len(mseq) / self.prefs[
                            ASSOC_ERROR_RATE]
                if reeval:
                    if mol in self.associations:
                        aseqs = [self.associations[mol],
                                    seq]
                    else:
                        aseqs = [seq]
                for aseq in aseqs:
                    if bestErrors:
                        tryErrors = bestErrors - 1
                    else:
                        tryErrors = maxErrors
                    try:
                        match_map, errors = tryAssoc(
                            aseq, mseq, segments,
                            gaps, estLen,
                            maxErrors=tryErrors)
                    except ValueError:
                        # maybe the sequence is
                        # derived from the structure...
                        if gaps:
                            try:
                                match_map, \
                                errors = \
                                tryAssoc(aseq,
                                mseq, [mseq[:]],
                                [], len(mseq),
                                maxErrors=
                                tryErrors)
                            except ValueError:
                                continue
                        else:
                            continue
                    else:
                        # if the above worked but
                        # had errors, see if just
                        # smooshing sequence together
                        # works better
                        if errors and gaps:
                            try:
                                match_map, \
                                errors = \
                                tryAssoc(aseq,
                                mseq, [mseq[:]],
                                [], len(mseq),
                                maxErrors=
                                errors-1)
                            except ValueError:
                                pass

                    bestMatchMap = match_map
                    bestErrors = errors
                    bestSeq = aseq
                    if errors == 0:
                        break

                if bestSeq:
                    if assocInfo \
                    and bestErrors >= assocInfo[-1]:
                        continue
                    assocInfo = (bestSeq, mseq,
                        bestMatchMap, bestErrors)
            if not assocInfo and force:
                # nothing matched built-in criteria
                # use Needleman-Wunsch
                bestSeq = bestMseq = bestErrors = None
                maxErrors = len(mseq) / self.prefs[
                            ASSOC_ERROR_RATE]
                for mseq in mseqs:

                    # aseqs are already sorted by length...
                    for aseq in aseqs:
                        self.status(
        "Using Needleman-Wunsch to test-associate %s %s with %s\n"
                            % (molName,
                            mseq.name, aseq.name))
                        match_map, errors = nwAssoc(
                                aseq, mseq)
                        if not bestSeq \
                        or errors < bestErrors:
                            bestMatchMap = match_map
                            bestErrors = errors
                            bestSeq = aseq
                            bestMseq = mseq
                if bestMatchMap:
                    assocInfo = (bestSeq, bestMseq,
                        bestMatchMap, bestErrors)
                else:
                    self.status("No reasonable association"
                        " found for %s %s\n" % (molName,
                        mseq.name))

            if assocInfo:
                bestSeq, mseq, bestMatchMap, bestErrors = \
                                assocInfo
                if reeval \
                and mseq.molecule in self.associations:
                    old_aseq = self.associations[
                                mseq.molecule]
                    if old_aseq == bestSeq:
                        continue
                    self.disassociate(mseq.molecule)
                msg = "Associated %s %s to %s with %d error(s)"\
                        "\n" % (molName, mseq.name,
                        bestSeq.name, bestErrors)
                self.status(msg, log=1, followWith=
                    "Right-click to focus on residue\n"
                    "Right-shift-click to focus on region",
                    followLog=False, blankAfter=10)
                self.prematched_assoc_structure(bestSeq, mseq,
                        bestMatchMap, bestErrors, reassoc)
                new_match_maps.append(bestMatchMap)
        if self.intrinsicStructure and len(self.seqs) == 1:
            self.showSS()
            self.status("Helices/strands depicted in gold/green")
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

    def prematched_assoc_structure(self, aseq, mseq, match_map, errors, reassoc):
        """If somehow you had obtained a match_map for the aseq<->mseq correspondence,
           you would use this call instead of the more usual associate() call
        """
        #TODO: support 'circular' attr in C++
        if getattr(aseq, 'circular', False):
            offset = len(aseq.ungapped())/2
            for k, v in match_map.items():
                if type(k) == chimera.Residue:
                    match_map[v + offset] = k
        mol = mseq.molecule
        # can have several 'chains' of the same sequence match to
        # one alignment sequence if there is an erroneous chain break
        match_map['mseq'] = mseq
        match_map['aseq'] = aseq
        try:
            aseq.match_maps[mol] = match_map
        except AttributeError:
            aseq.match_maps = { mol: match_map }
        self.associations[mol] = aseq

        self.seqCanvas.assocSeq(aseq)
        if hasattr(aseq, 'residueSequence'):
            aref = aseq.residueSequence
            errors = True
        else:
            aref = aseq.ungapped()
        # set up callbacks for structure changes
        match_map["mavDelHandler"] = mseq.triggers.addHandler(
                mseq.TRIG_DELETE, self._mseqDelCB, match_map)
        match_map["mavModHandler"] = mseq.triggers.addHandler(
                mseq.TRIG_MODIFY, self._mseqModCB, match_map)

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
