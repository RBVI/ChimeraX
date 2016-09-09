# vim: set expandtab shiftwidth=4 softtabstop=4:

def nw(s1, s2, score_match=10, score_mismatch=-3, score_gap=0, score_gap_open=-40,
            gap_char=".", return_seqs=False, score_matrix=None,
            similarity_matrix=None, frequency_matrix=None,
            ends_are_gaps=False, ss_matrix=None, ss_fraction=0.9,
            gap_open_helix=None, gap_open_strand=None,
            gap_open_other=None, debug=False):
    # if 'score_matrix', 'similarity_matrix', or 'frequency_matrix' is
    # provided, then 'score_match' and 'score_mismatch' are ignored and
    # the matrix is used to evaluate matching between the sequences.
    # 'score_matrix' should be a two-dimensional array of size
    # len(s1) x len(s2).  'similarity_matrix' should be a dictionary
    # keyed with two-tuples of residue types.  'frequency_matrix' should
    # be a list of length s2 of dictionaries, keyed by residue type.
    #
    # if 'ss_fraction' is not None/False, then 'ss_matrix' should be a 3x3
    # matrix keyed with 2-tuples of secondary structure types ('H': helix,
    # 'S': strand, 'O': other).  The score will be a mixture of the
    # ss/similarity matrix scores weighted by the ss_fraction
    # [ss_fraction * ss score + (1 - ss_fraction) * similarity score]
    #
    # if 'gap_open_helix/Strand/Other' is not None and 'ss_fraction' is not
    # None/False, then score_gap_open is ignored when an intra-helix/
    # intra-strand/other gap is opened and the appropriate penalty
    # is applied instead
    #
    # if 'return_seqs' is True, then instead of returning a match list
    # (a list of two-tuples) as the second value, a two-tuple of gapped
    # Sequences will be returned.  In both cases, the first return value
    # is the match score.
    m = []
    bt = []
    for i1 in range(len(s1) + 1):
        m.append((len(s2) + 1) * [ 0 ])
        bt.append((len(s2) + 1) * [None])
        bt[i1][0] = 1
        if ends_are_gaps and i1 > 0:
            m[i1][0] = score_gap_open + i1 * score_gap
    for i2 in range(len(s2) + 1):
        bt[0][i2] = 2
        if ends_are_gaps and i2 > 0:
            m[0][i2] = score_gap_open * i2 * score_gap

    if similarity_matrix is not None:
        evaluate = lambda i1, i2: similarity_matrix[(s1[i1], s2[i2])]
    elif score_matrix is not None:
        evaluate = lambda i1, i2: score_matrix[i1][i2]
    elif frequency_matrix is not None:
        evaluate = lambda i1, i2: frequency_matrix[i2][s1[i1]]
    else:
        def evaluate(i1, i2):
            if s1[i1] == s2[i2]:
                return score_match
            return score_mismatch
    doing_ss =  ss_fraction is not None and ss_fraction is not False \
                        and ss_matrix is not None
    if doing_ss:
        prev_eval = evaluate
        sim_fraction = 1.0 - ss_fraction
        def ss_eval(i1, i2):
            if hasattr(s1, 'ss_freqs'):
                freqs1 = s1.ss_freqs[i1]
            else:
                freqs1 = {s1.ss_type(i1): 1.0}
            if hasattr(s2, 'ss_freqs'):
                freqs2 = s2.ss_freqs[i2]
            else:
                freqs2 = {s2.ss_type(i2): 1.0}
            val = 0.0
            for ss1, freq1 in freqs1.items():
                if ss1 == None:
                    continue
                for ss2, freq2 in freqs2.items():
                    if ss2 == None:
                        continue
                    val += freq1 * freq2 * ss_matrix[(ss1, ss2)]
            return val
        evaluate = lambda i1, i2: ss_fraction * ss_eval(i1, i2) + \
                    sim_fraction * prev_eval(i1, i2)

    # precompute appropriate gap-open penalties
    gap_open_1 = [score_gap_open] * (len(s1)+1)
    gap_open_2 = [score_gap_open] * (len(s2)+1)
    if ends_are_gaps:
        if gap_open_other is not None:
            gap_open_1[0] = gap_open_2[0] = gap_open_other
    else:
            gap_open_1[0] = gap_open_2[0] = 0
    if doing_ss and gap_open_other != None:
        for seq, gap_opens in [(s1, gap_open_1), (s2, gap_open_2)]:
            if hasattr(seq, 'gap_freqs'):
                for i, gap_freq in enumerate(seq.gap_freqs):
                    gap_opens[i+1] = \
                        gap_freq['H'] * gap_open_helix + \
                        gap_freq['S'] * gap_open_strand + \
                        gap_freq['O'] * gap_open_other
            else:
                ss_type = [seq.ss_type(i)
                        for i in range(len(seq))]
                for i, ss in enumerate(ss_type[:-1]):
                    nextSS = ss_type[i+1]
                    if ss == nextSS and ss == 'H':
                        gap_opens[i+1] = gap_open_helix
                    elif ss == nextSS and ss == 'S':
                        gap_opens[i+1] = gap_open_strand
                    else:
                        gap_opens[i+1] = gap_open_other

    col_gap_starts = [0] * len(s2) # don't care about column zero
    for i1 in range(len(s1)):
        row_gap_pos = 0
        for i2 in range(len(s2)):
            best = m[i1][i2] + evaluate(i1, i2)
            bt_type = 0
            if i2 + 1 < len(s2) or ends_are_gaps:
                col_gap_pos = col_gap_starts[i2]
                skip_size = i1 + 1 - col_gap_pos
                if hasattr(s1, "occupancy"):
                    tot_occ = 0.0
                    for i in range(col_gap_pos, i1+1):
                        tot_occ += s1.occupancy[i]
                    col_skip_val = tot_occ * score_gap
                else:
                    col_skip_val = skip_size * score_gap
                base_col_gap_val = m[col_gap_pos][i2+1] + col_skip_val
                skip = base_col_gap_val + gap_open_2[i2+1]
            else:
                skip_size = 1
                col_skip_val = 0
                skip = m[i1][i2+1]
            if skip > best:
                best = skip
                bt_type = skip_size
            if i1 + 1 < len(s1) or ends_are_gaps:
                skip_size = i2 + 1 - row_gap_pos
                if hasattr(s2, "occupancy"):
                    tot_occ = 0.0
                    for i in range(row_gap_pos, i2+1):
                        tot_occ += s2.occupancy[i]
                    row_skip_val = tot_occ * score_gap
                else:
                    row_skip_val = skip_size * score_gap
                base_row_gap_val = m[i1+1][row_gap_pos] + row_skip_val
                skip = base_row_gap_val + gap_open_1[i1+1]
            else:
                skip_size = 1
                row_skip_val = 0
                skip = m[i1+1][i2]
            if skip > best:
                best = skip
                bt_type = 0 - skip_size
            m[i1+1][i2+1] = best
            bt[i1+1][i2+1] = bt_type
            if bt_type >= 0:
                # not gapping the row
                if best > base_row_gap_val:
                    row_gap_pos = i2 + 1
            if bt_type <= 0:
                # not gapping the column
                if best > base_col_gap_val:
                    col_gap_starts[i2] = i1 + 1
    """
    if debug:
        from chimera.selection import currentResidues
        cr = currentResidues(asDict=True)
        if cr:
            for fileName, matrix in [("scores", m), ("trace", bt)]:
                out = open("/home/socr/a/pett/rm/" + fileName,
                                    "w")
                print>>out, "    ",
                for i2, r2 in enumerate(s2.residues):
                    if r2 not in cr:
                        continue
                    print>>out, "%5d" % i2,
                print>>out
                print>>out, "    ",
                for i2, r2 in enumerate(s2.residues):
                    if r2 not in cr:
                        continue
                    print>>out, "%5s" % s2[i2],
                print>>out
                for i1, r1 in enumerate(s1.residues):
                    if r1 not in cr:
                        continue
                    print>>out, "%3d" % i1, s1[i1],
                    for i2, r2 in enumerate(s2.residues):
                        if r2 not in cr:
                            continue
                        print>>out, "%5g" % (
                            matrix[i1+1][i2+1]),
                    print>>out
                out.close()
    """
    i1 = len(s1)
    i2 = len(s2)
    match_list = []
    while i1 > 0 and i2 > 0:
        bt_type = bt[i1][i2]
        if bt_type == 0:
            match_list.append((i1-1, i2-1))
            i1 = i1 - 1
            i2 = i2 - 1
        elif bt_type > 0:
            i1 = i1 - bt_type
        else:
            i2 = i2 + bt_type
    if return_seqs:
        return m[len(s1)][len(s2)], matches_to_gapped_seqs(match_list,
                            s1, s2, gap_char=gap_char)
    return m[len(s1)][len(s2)], match_list

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
