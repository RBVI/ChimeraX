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
_GapChars = "-. "

from abc import ABC, abstractmethod
import json
from typing import Callable

class Parser(ABC):
    """Abstract base class for BLAST JSON parsers. To define a parser for a new
    type of database, create a subclass that implements _extract_hit"""
    def __init__(self, query_title, query_seq, output):
        self.true_name = query_title
        self.query_seq = query_seq
        self.output = output
        self._parse()

    def _parse(self) -> None:
        """
        extract_hit: A function that will parse the hits from BLAST output.
        """
        # Bookkeeping data
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
        self.res = json.loads(self.output)

        if not 'BlastOutput2' in self.res.keys():
            raise ValueError("Text is not BLAST JSON output")

        self.res_data = self.res["BlastOutput2"][0]["report"]
        self._extract_metadata(self.res_data)

        e = self.res_data["params"]
        if e is not None:
            self._extract_params(e)

        num_results = len(self.res_data["results"])
        if num_results > 1:
            raise ValueError("Multi-iteration BLAST output unsupported")
        elif num_results == 0:
            raise ValueError("No iteration data in BLAST output")
        for hit in self.res_data["results"]["search"]["hits"]:
            self._extract_hit(hit)
        self._extract_stats(self.res_data["results"]["search"]["stat"])
        self._append_query()

    @abstractmethod
    def _extract_hit(self, hit):
        pass

    def _append_query(self):
        """Insert the query as the first match"""
        m = Match(self.true_name, None, "user_input",
                  0.0, 0.0, 1, len(self.query_seq), self.query_seq, self.query_seq) #SH
        self.matches.insert(0, m)
        self.match_dict[self.query] = m

        # Go back and fix up hit sequences so that they all align
        # with the query sequence
        self._align_sequences()

    def _extract_metadata(self, md):
        self.database = md["search_target"]["db"]
        self.query = md["results"]["search"]["query_id"]
        self.query_length = md["results"]["search"]["query_len"]
        self._gap_count = [ 0 ] * self.query_length
        self.reference = md["reference"]
        self.version = md["version"]

    def _extract_params(self, pe):
        self.gap_existence = pe["gap_open"]
        self.gap_extension = pe["gap_extend"]
        self.matrix = pe["matrix"]

    def _extract_stats(self, sts):
        self.db_size_sequences = sts["db_num"]
        self.db_size_letters = sts["db_len"]

    def _extract_hsp(self, hsp, name, match_id, desc):
        score = int(float(hsp["bit_score"]))
        evalue = float(hsp["evalue"])
        h_seq = hsp["hseq"]
        h_start = int(hsp["hit_from"])
        h_end = int(hsp["hit_to"])
        q_seq = hsp["qseq"]
        q_start = int(hsp["query_from"])
        q_end = int(hsp["query_to"])
        self._update_gap_counts(q_seq, q_start, q_end)
        m = Match(name, match_id, desc, score, evalue, q_start, q_end, q_seq, h_seq)
        self.match_dict[name] = m
        return m

    def _copy_match(self, m, name, pdb, desc):
        nm = Match(name, pdb, desc, m.score, m.evalue,
                   m.q_start + 1, m.q_end + 1, # switch back to 1-base indexing
                   m.q_seq, m.h_seq)
        self.matches.append(nm)
        self.match_dict[name] = nm

    def _update_gap_counts(self, seq, start, end):
        start -= 1 # Switch to 0-based indexing
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
                print("  %s: %s" % (a, attr), file=f)
            elif isinstance(attr, list):
                for o in attr:
                    o.dump(f)
            elif attr is None:
                print("  %s: _uninitialized_" % a, file=f)


class PDBParser(Parser):
    def __init__(self, query_title, query_seq, output):
        super().__init__(query_title, query_seq, output)

    @staticmethod
    def _get_id_info(pdb_id, chain_id, hit_desc):
        name = None
        pdb = None
        desc = None
        if chain_id == "":
            name = pdb_id
            pdb = None
        else:
            name = pdb = '_'.join([pdb_id, chain_id])
        desc = hit_desc
        return name, pdb, desc

    def _extract_hit(self, hit):
        id_list = []
        # Unlike XML output, JSON output doesn't separate out the first PDBID.
        for entry in hit["description"]:
            if entry["id"].startswith("pdb"):
                _, name, chain = entry["id"].split("|")
            else:
                name, chain = entry["accession"], ""
            desc = entry["title"]
            if desc.startswith("Chain"):
                # Strip the chain information up to the first comma, but since
                # the description can have many commas splice the description
                # back together at the end
                desc = (','.join(desc.split(',')[1:])).strip()
            id_list.append((name, chain, desc))
        name, pdb, desc = PDBParser._get_id_info(*id_list[0])
        match_list = []
        for hsp in hit["hsps"]:
            match_list.append(self._extract_hsp(hsp, name, pdb, desc))
        for pdbid, chain, desc in id_list:
            name, pdb, _ = PDBParser._get_id_info(pdbid, chain, desc)
            for m in match_list:
                self._copy_match(m, name, pdb, desc)


class AlphaFoldParser(Parser):
    def __init__(self, query_title, query_seq, output):
        super().__init__(query_title, query_seq, output)

    def _extract_hit(self, hit):
        id_list = []
        for entry in hit["description"]:
            uniprot_id = entry["title"].split('|')[1]
            desc = entry["title"].split('|')[2]
            id_list.append((uniprot_id, desc))
        match_list = []
        for hsp in hit["hsps"]:
            match_list.append(self._extract_hsp(hsp, uniprot_id, uniprot_id, desc))
        for uniprot_id, desc in id_list:
            for m in match_list:
                uniprot_name = desc.split('=')[0].split(' ')[0].split('|')[-1]
                self._copy_match(m, uniprot_name, uniprot_name, desc)


class Match:
    """Data from a single BLAST hit."""

    def __init__(self, name, match_id, desc, score, evalue,
                 q_start, q_end, q_seq, h_seq):
        self.name = name
        self.match = match_id
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
        return "<Match %s (match=%s)>" % (self.name, self.match)

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
        print(self, file=f)
        self.print_sequence(f, '')
