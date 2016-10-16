# vim: set expandtab shiftwidth=4 softtabstop=4:

_GapChars = "-. "

import re
RE_PDBId = re.compile(r"\S*pdb\|(?P<id>\w{4})\|(?P<chain>\w*)\s*(?P<desc>.*)")

class Parser:
    """Parser for XML output from blastp (tested against
    version 2.2.29+."""

    def __init__(self, true_name, query_seq, output):
        # Bookkeeping data
        self.true_name = true_name
        self.matches = []
        self.match_dict = {}
        self._gap_count = None

        # Data from results
        self.database = None
        self.query = None
        self.query_length = None
        self.reference = None
        self.version = None

        self.gap_existence = None
        self.gap_extension = None
        self.matrix = None

        self.db_size_sequences = None
        self.db_size_letters = None

        # Extract information from results
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(output)
        if tree.tag != "BlastOutput":
            raise ValueError("Text is not BLAST XML output")
        self._extract_root(tree)
        e = tree.find("./BlastOutput_param/Parameters")
        if e is not None:
            self._extract_params(e)
        el = tree.findall("BlastOutput_iterations/Iteration")
        if len(el) > 1:
            raise ValueError("Multi-iteration BLAST output unsupported")
        elif len(el) == 0:
            raise ValueError("No iteration data in BLAST OUTPUT")
        iteration = el[0]
        for he in iteration.findall("./Iteration_hits/Hit"):
            self._extract_hit(he)
        self._extract_stats(iteration.find("./Iteration_stat/Statistics"))

        # Insert the query as match[0]
        m = Match(self.true_name, None, "user_input",
                  0.0, 0.0, 1, len(query_seq), query_seq, query_seq) #SH
        self.matches.insert(0, m)
        self.match_dict[self.query] = m

        # Go back and fix up hit sequences so that they all align
        # with the query sequence
        self._align_sequences()

    def _text(self, parent, tag):
        e = parent.find(tag)
        return e is not None and e.text.strip() or None

    def _extract_root(self, oe):
        self.database = self._text(oe, "BlastOutput_db")
        self.query = self._text(oe, "BlastOutput_query-ID")
        self.query_length = int(self._text(oe, "BlastOutput_query-len"))
        self._gap_count = [ 0 ] * self.query_length
        self.reference = self._text(oe, "BlastOutput_reference")
        self.version = self._text(oe, "BlastOutput_version")

    def _extract_params(self, pe):
        self.gap_existence = self._text(pe, "Parameters_gap-open")
        self.gap_extension = self._text(pe, "Parameters_gap-extend")
        self.matrix = self._text(pe, "Parameters_matrix")

    def _extract_stats(self, se):
        self.db_size_sequences = self._text(se, "Statistics_db-num")
        self.db_size_letters = self._text(se, "Statistics_db-len")

    def _extract_hit(self, he):
        hid = self._text(he, "Hit_id")
        m = RE_PDBId.match(hid)
        if m:
            # PDB hit, create list of PDB hits
            id_list = []
            for defline in (hid + ' ' + self._text(he, "Hit_def")).split(">"):
                m = RE_PDBId.match(defline)
                if m:
                    id_list.append(m.groups())
            pdbid, chain, desc = id_list.pop(0)
            name = pdb = pdbid + '_' + chain if chain else pdbid
        else:
            name = hid
            pdb = None
            desc = self._text(he, "Hit_def").split(">")[0]
            # An nr hit can have many more ids on the defline, but
            # we only keep pdb ones
            id_list = []
            for defline in (hid + ' ' + self._text(he, "Hit_def")).split(">"):
                m = RE_PDBId.match(defline)
                if m:
                    id_list.append(m.groups())
        match_list = []
        for hspe in he.findall("./Hit_hsps/Hsp"):
            match_list.append(self._extract_hsp(hspe, name, pdb, desc))
        for pdbid, chain, desc in id_list:
            name = pdb = pdbid + '_' + chain if chain else pdbid
            for m in match_list:
                self._copy_match(m, name, pdb, desc)

    def _extract_hsp(self, hspe, name, pdb, desc):
        score = int(float(self._text(hspe, "Hsp_bit-score"))) #SH
        evalue = float(self._text(hspe, "Hsp_evalue"))
        q_seq = self._text(hspe, "Hsp_qseq")
        q_start = int(self._text(hspe, "Hsp_query-from"))
        q_end = int(self._text(hspe, "Hsp_query-to"))
        self._update_gap_counts(q_seq, q_start, q_end)
        h_seq = self._text(hspe, "Hsp_hseq")
        h_start = int(self._text(hspe, "Hsp_hit-from"))
        h_end = int(self._text(hspe, "Hsp_hit-to"))
        m = Match(name, pdb, desc, score, evalue, q_start, q_end, q_seq, h_seq)
        self.matches.append(m)
        self.match_dict[name] = m
        return m

    def _copy_match(self, m, name, pdb, desc):
        nm = Match(name, pdb, desc, m.score, m.evalue,
                   m.q_start + 1, m.q_end + 1, # switch back to 1-base indexing
                   m.q_seq, m.h_seq)
        self.matches.append(nm)
        self.match_dict[name] = nm

    def _update_gap_counts(self, seq, start, end):
        start -= 1    # Switch to 0-based indexing
        count = 0
        for c in seq:
            if c in _GapChars:
                count += 1
            else:
                old_count = self._gap_count[start]
                self._gap_count[start] = max(old_count, count)
                start += 1
                count = 0

    def _align_sequences(self):
        for m in self.matches:
            m.match_sequence_gaps(self._gap_count)

    def write_msf(self, f, per_line=60, block=10, matches=None):
        if (matches is not None and len(matches) == 1 and
            matches[0] is self.matches[0]):
            # if user selected only the query sequence,
            # we treat it as if he selected nothing at all
            matches = None
        if matches is None:
            matches = self.matches
        if self.matches[0] not in matches:
            matches.insert(0, self.matches[0])
        length = len(matches[0].sequence)
        # Assumes that all sequence lengths are equal

        f.write("Query: %s\n" % self.query)
        f.write("BLAST Version: %s\n" % self.version)
        f.write("Reference: %s\n" % self.reference)
        f.write("Database: %s\n" % self.database)
        f.write("Database size: %s sequences, %s letters\n" %
            (self.db_size_sequences, self.db_size_letters))
        f.write("Matrix: %s\n" % self.matrix)
        f.write("Gap penalties: existence: %s, extension: %s\n" %
            (self.gap_existence, self.gap_extension))
        f.write("\n")
        label = {}
        for m in matches:
            label[m] = m.name
        width = max(map(lambda m: len(label[m]), matches[1:]))
        for m in matches[1:]:
            f.write("%*s %4d %g" %
                (width, label[m], m.score, m.evalue))
            if m.description:
                f.write(" %s\n" % m.description)
            else:
                f.write("\n")
        f.write("\n")

        import time
        now = time.strftime("%B %d, %Y %H:%M",
                    time.localtime(time.time()))
        f.write(" %s  MSF: %d  Type: %s  %s  Check: %d ..\n\n"
                % ("BLAST", length, 'P', now , 0))

        name_width = max(map(lambda m: len(label[m]), matches))
        name_fmt = " Name: %-*s  Len: %5d  Check: %4d  Weight: %5.2f\n"
        for m in matches:
            f.write(name_fmt % (name_width, label[m], length, 0, 1.0))
        f.write("\n//\n\n")

        for i in range(0, length, per_line):
            start = i + 1
            end = start + per_line - 1
            if end > length:
                end = length
            seq_len = end - start + 1
            start_label = str(start)
            end_label = str(end)
            separators = (seq_len + block - 1) / block - 1
            blanks = (seq_len + separators
                    - len(start_label) - len(end_label))
            if blanks < 0:
                f.write("%*s  %s\n" %
                    (name_width, ' ', start_label))
            else:
                f.write("%*s  %s%*s%s\n" % (name_width, ' ', start_label,
                                            blanks, ' ', end_label))
            for m in matches:
                f.write("%-*s " % (name_width, label[m]))
                for n in range(0, per_line, block):
                    front = i + n
                    back = front + block
                    f.write(" %s" % m.sequence[front:back])
                f.write("\n")
            f.write("\n")

    def session_data(self):
        try:
            from cPickle import dumps
        except ImportError:
            from pickle import dumps
        return dumps(self)

    def dump(self, f=None):
        if f is None:
            from sys import stderr as f
        for a in dir(self):
            if a.startswith("_"):
                continue
            attr = getattr(self, a)
            if callable(attr):
                continue
            if isinstance(attr, basestring):
                print >> f, "  %s: %s" % (a, attr)
            elif isinstance(attr, list):
                for o in attr:
                    o.dump(f)
            elif attr is None:
                print >> f, "  %s: _uninitialized_" % a

def restore_parser(data):
    try:
        from cPickle import loads
    except ImportError:
        from pickle import loads
    return loads(data)

class Match:
    """Data from a single BLAST hit."""

    def __init__(self, name, pdb, desc, score, evalue,
                 q_start, q_end, q_seq, h_seq):
        self.name = name
        self.pdb = pdb
        self.description = desc.strip()
        self.score = score
        self.evalue = evalue
        self.q_start = q_start - 1  # switch to 0-base indexing
        self.q_end = q_end - 1      # switch to 0-base indexing
        self.q_seq = q_seq
        self.h_seq = h_seq
        if len(q_seq) != len(h_seq):
            raise ValueError("sequence alignment length mismatch")
        self.sequence = ""

    def __repr__(self):
        return "<Match %s (pdb=%s)>" % (self.name, self.pdb)

    def print_sequence(self, f, prefix, per_line=60):
        for i in range(0, len(self.sequence), per_line):
            f.write("%s%s\n" % (prefix, self.sequence[i:i+per_line]))

    def match_sequence_gaps(self, gap_count):
        seq = []
        # Insert gap for head of query sequence that did not match
        for i in range(self.q_start):
            seq.append('.' * (gap_count[i] + 1))
        start = self.q_start
        count = 0
        # Add all the sequence data from this HSP
        for i in range(len(self.q_seq)):
            if self.q_seq[i] in _GapChars:
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
        print >> f, self
        self.print_sequence(f, '')
