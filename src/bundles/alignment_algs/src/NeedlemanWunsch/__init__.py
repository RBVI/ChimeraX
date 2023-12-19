# vim: set expandtab shiftwidth=4 softtabstop=4:

def nw(s1, s2, score_match=10, score_mismatch=-3, score_gap=0, score_gap_open=-40,
            gap_char=".", return_seqs=False, score_matrix=None,
            similarity_matrix=None, frequency_matrix=None,
            ends_are_gaps=False, ss_matrix=None, ss_fraction=0.9,
            gap_open_helix=None, gap_open_strand=None,
            gap_open_other=None, debug=False):
    """Compute Needleman-Wunsch alignment

    if 'score_matrix', 'similarity_matrix', or 'frequency_matrix' is
    provided, then 'score_match' and 'score_mismatch' are ignored and
    the matrix is used to evaluate matching between the sequences.
    'score_matrix' should be a two-dimensional array of size
    len(s1) x len(s2).  'similarity_matrix' should be a dictionary
    keyed with two-tuples of residue types.  'frequency_matrix' should
    be a list of length s2 of dictionaries, keyed by residue type.
    
    if 'ss_fraction' is not None/False, then 'ss_matrix' should be a 3x3
    matrix keyed with 2-tuples of secondary structure types ('H': helix,
    'S': strand, 'O': other).  The score will be a mixture of the
    ss/similarity matrix scores weighted by the ss_fraction
    [ss_fraction * ss score + (1 - ss_fraction) * similarity score]
    
    if 'gap_open_helix/Strand/Other' is not None and 'ss_fraction' is not
    None/False, then score_gap_open is ignored when an intra-helix/
    intra-strand/other gap is opened and the appropriate penalty
    is applied instead
    
    if 'return_seqs' is True, then instead of returning a match list
    (a list of two-tuples) as the second value, a two-tuple of gapped
    Sequences will be returned.  In both cases, the first return value
    is the match score."""

    # Make sure _nw can runtime link shared library libarrays.
    import chimerax.arrays
    from .._nw import match
    if gap_open_helix is None:
        ss_specific_gaps = False
        go_helix = go_strand = go_other = 0.0
    else:
        ss_specific_gaps = True
        go_helix = gap_open_helix
        go_strand = gap_open_strand
        go_other = gap_open_other
    score, match_list = match(s1.characters, s2.characters, score_match, score_mismatch,
        score_gap_open, score_gap, ends_are_gaps, similarity_matrix, score_matrix,
        frequency_matrix, ss_matrix, ss_fraction, ss_specific_gaps, go_helix, go_strand,
        go_other, getattr(s1, 'gap_freqs', None), getattr(s2, 'gap_freqs', None),
        "".join([s1.ss_type(i) or ' ' for i in range(len(s1))]),
        "".join([s2.ss_type(i)  or ' 'for i in range(len(s2))]),
        getattr(s1, 'ss_freqs', None), getattr(s2, 'ss_freqs', None),
        getattr(s1, 'occupancy', None), getattr(s2, 'occupancy', None))
    if return_seqs:
        return score, matches_to_gapped_seqs(match_list, s1, s2, gap_char=gap_char)
    return score, match_list

def matches_to_gapped_seqs(matches, s1, s2, gap_char=".", reverse_sorts=True):
    gapped1 = clone_seq(s1)
    gapped2 = clone_seq(s2)
    prev1 = prev2 = -1
    if reverse_sorts:
        matches.reverse()
    else:
        matches.sort()
    for pos1, pos2 in matches:
        if pos1 > prev1 + 1 and pos2 > prev2 + 1:
            gapped1.extend(s1[prev1+1:pos1])
            gapped2.extend(gap_char * (pos1 - prev1 - 1))
            gapped1.extend(gap_char * (pos2 - prev2 - 1))
            gapped1.append(s1[pos1])
            gapped2.extend(s2[prev2+1:pos2+1])
        else:
            if pos1 > prev1 + 1:
                gapped1.extend(s1[prev1+1:pos1+1])
                gapped2.extend(gap_char * (pos1 - prev1 - 1))
                gapped2.append(s2[pos2])
            if pos2 > prev2 + 1:
                gapped1.extend(gap_char * (pos2 - prev2 - 1))
                gapped1.append(s1[pos1])
                gapped2.extend(s2[prev2+1:pos2+1])
            if pos1 == prev1 + 1 and pos2 == prev2 + 1:
                gapped1.append(s1[pos1])
                gapped2.append(s2[pos2])
        prev1, prev2 = pos1, pos2
    if prev1 < len(s1) - 1:
        gapped2.extend(gap_char * (len(s1) - prev1 - 1))
        gapped1.extend(s1[prev1+1:])
    if prev2 < len(s2) - 1:
        gapped1.extend(gap_char * (len(s2) - prev2 - 1))
        gapped2.extend(s2[prev2+1:])
    return gapped1, gapped2

def clone_seq(seq):
    from copy import copy
    clone = copy(seq)
    if hasattr(clone, "structure"):
        clone.name = clone.structure.name + ", " + clone.name
    clone[:] = ""
    return clone
