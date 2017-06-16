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

from time import time
ssvt = 0
def segmentSieve(rList0, rList1, fraction=0.5):
        """MolMovDB style segmenting.  Use sieve fit to find core.
        Everything else is in the second segment."""
        # sieve_fit checks that the two molecule have equivalent residues
        t0 = time()
        from . import sieve_fit
        coreList0, coreList1 = sieve_fit.fitResidues(rList0, rList1, fraction)
        def splitCore(rList, coreSet):
                core = []
                others = []
                for r in rList:
                        if r in coreSet:
                                core.append(r)
                                r._in_morph_core = True  # Flag residues in core for later coloring.
                        else:
                                others.append(r)
                return core, others

        # First we split our residue list by chain identifier
        # For each chain, we split into a core segment and an other segment
        coreSet0 = set(coreList0)
        coreSet1 = set(coreList1)
        segments = list()
        start = 0
        chain = rList0[0].chain_id
        for i in range(1, len(rList0)):
                r0 = rList0[i]
                if r0.chain_id != chain:
                        core0, others0 = splitCore(rList0[start:i], coreSet0)
                        core1, others1 = splitCore(rList1[start:i], coreSet1)
                        segments.append((core0, core1))
                        segments.append((others0, others1))
                        start = i
        if start < len(rList0):
                core0, others0 = splitCore(rList0[start:], coreSet0)
                core1, others1 = splitCore(rList1[start:], coreSet1)
                segments.append((core0, core1))
                segments.append((others0, others1))
        t1 = time()
        global ssvt
        ssvt += t1-t0
        return segments

# Require same chain id, same residue number and same atom name for pairing.
def segmentHingeSame(m0, m1, fraction=0.5, min_hinge_spacing=6):

        # Split by chain
        cr0, cr1 = m0.residues.by_chain, m1.residues.by_chain
        cids = set(cid for s,cid,rlist in cr0) & set(cid for s,cid,rlist in cr1)
        cr0 = {cid:rlist for s,cid,rlist in cr0 if cid in cids}
        cr1 = {cid:rlist for s,cid,rlist in cr1 if cid in cids}

        parts = []
        atomMap = {}
        for cid in cids:
                r0list, r1list = cr0[cid], cr1[cid]
                r0map = {r.number:r for r in r0list}
                r1map = {r.number:r for r in r1list}
                rnums = set(r0map.keys()) & set(r1map.keys())
                curRList0 = []
                curRList1 = []
                for rnum in rnums:
                        r0, r1 = r0map[rnum], r1map[rnum]
                        if shareAtomsSameName(r0, r1, atomMap):
                                curRList0.append(r0)
                                curRList1.append(r1)
                parts.append((curRList0, curRList1))

        #
        # Split each part on hinges and collate results
        #
        segments = []
        for rList0, rList1 in parts:
                segments.extend(segmentHingeResidues(rList0, rList1, fraction, min_hinge_spacing))
        return segments, atomMap

def segmentHingeExact(m0, m1, fraction=0.5, min_hinge_spacing=6):

        # Split by chain
        cr0, cr1 = m0.residues.by_chain, m1.residues.by_chain
        if len(cr0) != len(cr1):
                raise ValueError("models have different number of chains")

        # If chain ids all match then pair same ids, otherwise use order from file.
        if set(cid for s,cid,rlist in cr0) == set(cid for s,cid, rlist in cr1):
                cr0.sort(key = lambda scr: scr[1])
                cr1.sort(key = lambda scr: scr[1])

        parts = []
        atomMap = {}
        for (s0,cid0,r0list), (s1,cid1,r1list) in zip(cr0, cr1):
                if len(r0list) != len(r1list):
                        raise ValueError("Chains %s and %s have different number of residues" % (cid0, cid1))
                curCat = None
                curRList0 = None
                curRList1 = None
                for r0, r1 in zip(r0list, r1list):
                        if not shareAtoms(r0, r1, atomMap):
                                raise ValueError("residues do not share atoms")
                        #
                        # Split residues based on surface category (in m0)
                        #
                        cat = r0.atoms[0].structure_category
                        if cat == curCat:
                                curRList0.append(r0)
                                curRList1.append(r1)
                        else:
                                curRList0 = [ r0 ]
                                curRList1 = [ r1 ]
                                parts.append((curRList0, curRList1))
                                curCat = cat

        #
        # Split each part on hinges and collate results
        #
        segments = []
        for rList0, rList1 in parts:
                segments.extend(segmentHingeResidues(rList0, rList1, fraction, min_hinge_spacing))
        return segments, atomMap

def segmentHingeApproximate(m0, m1, fraction=0.5, min_hinge_spacing=6, matrix="BLOSUM-62"):
        #
        # Get the chains from each model.  If they do not have the
        # same number of chains, we give up.  Otherwise, we assume
        # that the chains should be matched in the same order.
        #
        m0seqs = m0.chains
        m1seqs = m1.chains
        if len(m0seqs) != len(m1seqs):
                raise ValueError("models have different number of chains")
        resCount0 = len(m0.residues)
        matchCount0 = 0
        for seq0 in m0seqs:
                matchCount0 += len(seq0.residues)
#        print ("Aligning %d of %d residues from molecule %s" % (
#                        matchCount0, resCount0, m0.name))
        resCount1 = len(m1.residues)
        matchCount1 = 0
        for seq1 in m1seqs:
                matchCount1 += len(seq1.residues)
#        print ("Aligning %d of %d residues from molecule %s" % (
#                        matchCount1, resCount1, m1.name))

        #
        # Try to find the best matches for sequences.
        # If both models have chains with the same ids, assume that
        # chains with the same ids match.  Otherwise, assume the chains
        # match in input order.
        #
        m0map = dict((s.chain, s) for s in m0seqs)
        m1map = dict((s.chain, s) for s in m1seqs)
        if set(m0map.keys()) == set(m1map.keys()):
                seqPairs = [ (m0map[k], m1map[k]) for k in m0map.keys() ]
        else:
                seqPairs = zip(m0seqs, m1seqs)

        #
        # For each chain pair, we align them using MatchMaker to get
        # the residue correspondences.
        #
        from chimerax.seqalign.sim_matrices import matrix_compatible
        from chimerax.match_maker.settings import defaults
        from chimerax.match_maker.match import align
        ksdsspCache = set([m0, m1])
        parts = []
        atomMap = {}
        matched = 0
        matrices = [
                defaults['matrix'],
                "Nucleic",
        ]
        session = m0.session
        for seq0, seq1 in seqPairs:
                for matrix in matrices:
                        if (matrix_compatible(session, seq0, matrix)
                        and matrix_compatible(session, seq1, matrix)):
                                break
                else:
                        continue
                score, gapped0, gapped1 = align(session, seq0, seq1,
                                matrix, "nw",
                                defaults['gap_open'],
                                defaults['gap_extend'],
                                ksdsspCache)
                rList0 = []
                rList1 = []
                for pos in range(len(gapped0)):
                        i0 = gapped0.gapped_to_ungapped(pos)
                        if i0 is None:
                                continue
                        r0 = gapped0.residues[i0]
                        if r0 is None:
                                continue
                        i1 = gapped1.gapped_to_ungapped(pos)
                        if i1 is None:
                                continue
                        r1 = gapped1.residues[i1]
                        if r1 is None:
                                continue
                        if not shareAtoms(r0, r1, atomMap):
                                continue
                        rList0.append(r0)
                        rList1.append(r1)
                        matched += 1
                if rList0:
                        parts.append((rList0, rList1))

        #
        # Split each part on hinges and collate results
        #
        segments = []
        for rList0, rList1 in parts:
                segments.extend(segmentHingeResidues(rList0, rList1, fraction, min_hinge_spacing))

        #
        # Identify any residues that were not in sequences but have
        # similar connectivity (e.g., metals and ligands) and share
        # some common atoms
        #
        segmentMap = dict()
        residueMap = dict()
        for sIndex, s in enumerate(segments):
                for r0, r1 in zip(s[0], s[1]):
                        residueMap[r1] = r0
                        segmentMap[r0] = sIndex
        used = set()
        for seq0 in m0seqs:
                used.update(seq0.residues)
        m0candidates = [ r0 for r0 in m0.residues if r0 not in used ]
        keyMap = dict()
        for r0 in m0candidates:
                neighbors = _getConnectedResidues(r0)
                if not neighbors:
                        continue
                nlist = list(neighbors)
                nlist.sort()
                keyMap[tuple(nlist)] = r0
        used = set()
        for seq1 in m1seqs:
                used.update(seq1.residues)
        m1candidates = [ r1 for r1 in m1.residues if r1 not in used ]
        for r1 in m1candidates:
                neighbors = _getConnectedResidues(r1)
                if not neighbors:
                        continue
                try:
                        nlist = [ residueMap[r] for r in neighbors ]
                except KeyError:
                        pass
                else:
                        nlist.sort()
                        key = tuple(nlist)
                        try:
                                r0 = keyMap[key]
                                sIndex = segmentMap[nlist[0]]
                        except KeyError:
                                pass
                        else:
                                if shareAtoms(r0, r1, atomMap):
                                        s0, s1 = segments[sIndex]
                                        segments[sIndex] = (s0 + (r0,), s1 + (r1,))
                                        matched += 1

        #
        # Finally, finished
        #
        # print ("Matched %d residues in %d segments" % (matched, len(segments)))
        return segments, atomMap

def segmentHingeResidues(rList0, rList1, fraction, min_hinge_spacing):
        #
        # Find matching set of residues
        #
        segments = segmentSieve(rList0, rList1, fraction)

        #
        # Find hinges and split molecules at hinge residues
        # The Interpolate module wants a set of segments
        # which are 2-tuples.  Each element of the 2-tuple
        # is a tuple of residues.
        #
        from .hinge import findHinges, splitOnHinges
        hingeIndices = findHinges(rList0, rList1, segments, min_hinge_spacing)
        segmentsStart = [ tuple(l)
                        for l in splitOnHinges(hingeIndices, rList0) ]
        segmentsEnd = [ tuple(l)
                        for l in splitOnHinges(hingeIndices, rList1) ]
        segments = zip(segmentsStart, segmentsEnd)
        return segments

#
# Match atoms in r0 to atoms with the same name in r1 starting at
# the r0 atom that connects to the previous residue or lacking such an
# atom to the r0 atom that connects to the next residue, or lacking that
# atom start with the first r0 atom that has an atom of the same name in r1.
# Expand out along bonds from that starting atom matching atoms in r1 with
# same name and same bond to the base atom. This produces a matched
# subgraph with same atom names.  Then match any atoms that were not
# paired by name only.
#
# If a residue has more than one atom with the same name, the behavior
# of this matching will depend on the order of the atoms.  Actually
# it depends on the order of atoms in any case if bond patterns don't
# match.
#
# Only atoms with the same name will be matched, and some bonds must
# also match.
#
# Since we are primarily iterested in proteins and nucleic acids where
# all atoms have unique names, maybe we should just exclude pairing any
# atoms which have other atoms in the residue of the same name.  Will
# pairing everything else exactly by name cause problems?  Do we need
# to try to impose similar connectivity requirements?  I think standard
# amino acid and nucleic acid atom names imply connectivity.  So
# connectivity requirements get us nothing.  Might be different with
# small molecule ligands.
# 
satt = 0                        
def shareAtoms(r0, r1, atomMap):
        t0 = time()
        # We start by finding the atom connected to the
        # previous residue.  Failing that, we want the
        # atom connected to the next residue.  Failing that,
        # we take an arbitrary atom.
        c0 = r0.chain
        before = c0.residue_before(r0) if c0 else None
        after = c0.residue_after(r0) if c0 else None
        r0atoms = r0.atoms
        neighbors = {a:a.neighbors for a in r0atoms}
        startAtom = None
        for a0 in r0atoms:
                if startAtom is None:
                        startAtom = a0
                for na in neighbors[a0]:
                        if na.residue is before:
                                startAtom = a0
                                break
                        elif na.residue is after:
                                startAtom = a0
        # From this starting atom, we do a breadth-first
        # search for an atom with a matching atom name in r1
        todo = [ startAtom ]
        visited = set()
        a0 = a1 = None
        while todo:
                a0 = todo.pop(0)
                a1 = r1.find_atom(a0.name)
                if a1:
                        break
                else:
                        # No match, so we put all our neighboring
                        # atoms on the search list
                        visited.add(a0)
                        for na in neighbors[a0]:
                                if na not in visited and na in neighbors:
                                        todo.append(na)
        if a1 is None:
                return False

        matched = [(a0,a1)]
        paired = set((a0,a1))
        c = 0 
        while c < len(matched):
                a0, a1 = matched[c]
                c += 1
                # a0 and a1 are matched, now we want to see
                # if any of their neighbors match
                for na0 in neighbors[a0]:
                        if na0 not in paired and na0 in neighbors:
                                # Original Chimera 1 code checks all neighbors of na1.
                                # That allows for residues having atoms with the same name.
                                # If atoms have same name we pair with one that may be wrong
                                # and that will mess up all subsequent pairing.  But if we
                                # never find two atoms with same name attached then choice
                                # is unique.  For instance, long linear chain of connected carbons
                                # all named C could be handled.
                                na1 = r1.find_atom(na0.name)
                                if na1 and na1.connects_to(a1) and na1 not in paired:
                                        matched.append((na0, na1))
                                        paired.add(na0)
                                        paired.add(na1)

        # Next we check for atoms we have not visited and see if
        # we can pair them
        if len(matched) < len(r0atoms):
                for a0 in r0atoms:
                        if a0 in paired:
                                continue
                        a1 = r1.find_atom(a0.name)
                        if a1 is not None and a1 not in paired:
                                matched.append((a0, a1))
                                paired.add(a1)

        for a0,a1 in matched:
                atomMap[a0] = a1
        t1 = time()
        global satt
        satt += t1-t0
        return True

def shareAtomsSameName(r0, r1, atomMap):
        a1 = {a.name: a for a in r1.atoms}
        matched = False
        for a in r0.atoms:
                if a.name in a1:
                        atomMap[a] = a1[a.name]
                        matched = True
        return matched


def _getConnectedResidues(r):
        neighborResidues = set()
        for a in r.atoms:
                for na in a.neighbors:
                        if na.residue is not r:
                                neighborResidues.add(na.residue)
# TODO: Traverse pseudobonds.  Currently no atom.pseudoBonds property.
#                for pb in a.pseudoBonds:
#                        na = pb.otherAtom(a)
#                        if na.residue is not r:
#                                neighborResidues.add(na.residue)
        return neighborResidues
