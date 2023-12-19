# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from ..utils import SeqGapChars

class Match:
    """Data from a single BLAST hit."""

    def __init__(self, name, match_id, desc, score, evalue,
                 q_start, q_end, q_seq, h_seq, sequence = None):
        self.name = name
        self._match = match_id
        self.description = desc.strip()
        self._score = score
        self.evalue = evalue
        self.q_start = q_start - 1  # switch to 0-base indexing
        self.q_end = q_end - 1      # switch to 0-base indexing
        self.q_seq = q_seq
        self.h_seq = h_seq
        if len(q_seq) != len(h_seq):
            raise ValueError("sequence alignment length mismatch")
        if sequence is None:
            self.sequence = ""
        else:
            self.sequence = sequence

    @property
    def match(self):
        return self._match

    @property
    def score(self):
        return self._score

    def __eq__(self, other):
        return (
            (isinstance(other, Match))
            and (self.score == other.score)
            and (self.match == other.match)
        )

    def __hash__(self):
        return hash((self.score, self.match))

    def __repr__(self):
        return "<Match %s (match=%s)>" % (self.name, self.match)

    def print_sequence(self, f, prefix, per_line=60):
        for i in range(0, len(self.sequence), per_line):
            f.write("%s%s\n" % (prefix, self.sequence[i:i + per_line]))

    def match_sequence_gaps(self, gap_count):
        seq = []
        # Insert gap for head of query sequence that did not match
        for i in range(self.q_start):
            seq.append('.' * (gap_count[i] + 1))
        start = self.q_start
        count = 0
        # Add all the sequence data from this HSP
        for i in range(len(self.q_seq)):
            if self.q_seq[i] in SeqGapChars:
                # If this is a gap in the query sequence,
                # then the hit sequence must be an insertion.
                # Add the insertion to the final sequence
                # and increment number of gaps we have added
                # thus far.
                seq.append(self.h_seq[i])
                count += 1
            else:
                # If this is not a gap, then we have to make
                # sure that we have inserted enough gaps for
                # the longest insertion by any sequence (as
                # computed in "gap_count").  Then we add the
                # hit sequence character that matches this
                # query sequence character, and increment
                # out query sequence index ("start").
                if count > gap_count[start]:
                    print("start", start)
                    print("count", count, ">", gap_count[start])
                    raise ValueError("cannot align sequences")
                if count < gap_count[start]:
                    seq.append('-' * (gap_count[start] - count))
                seq.append(self.h_seq[i])
                count = 0
                start += 1
        # Append gap for tail of query sequence that did not match
        while start < len(gap_count):
            seq.append('.' * (gap_count[start] + 1))
            start += 1
        self.sequence = ''.join(seq)

    def dump(self, f):
        print(self, file=f)
        self.print_sequence(f, '')
