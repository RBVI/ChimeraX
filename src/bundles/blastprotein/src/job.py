# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from typing import Dict
from collections import defaultdict
from urllib3.exceptions import MaxRetryError

from chimerax.core.tasks import JobError
from chimerax.webservices.cxservices_job import CxServicesJob
from chimerax.atomic import Sequence

from .data_model import (
    get_database,
    CurrentDBVersions,
    parse_blast_results,
    AvailableDBsDict,
)
from .utils import BlastParams, make_instance_name


class BlastProteinJob(CxServicesJob):
    inet_error = "Could not start BLAST job. Please check your internet connection and try again."
    service_name = "blast"
    SESSION_SAVE = True

    def __init__(self, session, seq, atomspec, **kw):
        super().__init__(session)

        self.show_gui = kw.pop("show_gui", True)
        self.only_best = kw.pop("only_best", True)
        self.load_structures = kw.pop("load_structures", False)
        self.load_sequences = kw.pop("load_sequences", False)

        if "tool_inst_name" not in kw:
            kw["tool_inst_name"] = make_instance_name()
        if kw["tool_inst_name"] is None:
            kw["tool_inst_name"] = make_instance_name()

        try:
            self.setup(seq, atomspec, **kw)
        except JobError as e:
            session.logger.warning(" ".join(["Cannot submit job:", str(e)]))
            return

        self.params = {
            "db": self.database,
            "evalue": str(self.cutoff),
            "matrix": self.matrix,
            "blimit": str(self.max_seqs),
            "input_seq": self.seq,
            "version": self.version,
        }

        try:
            model_no = int(atomspec.split("/")[0].split("#")[1])
            chain = atomspec.split("/")[1]
            self.model_name = session.models._models[(model_no,)]._name
            self.chain = chain
        except (ValueError, KeyError, AttributeError):
            self.model_name = None

    def setup(
        self,
        seq,
        atomspec,
        database: str = "pdb",
        cutoff: float = 1.0e-3,
        matrix: str = "BLOSUM62",
        max_seqs: int = 100,
        log=None,
        version=None,
        tool_inst_name=None,
        sequence_name=None,
    ):
        self.seq = seq.replace("?", "X")  # string
        if self.seq.count("X") == len(self.seq):
            raise JobError("Sequence consists entirely of unknown amino acids.")
        # if self.seq.count('X') > len(self.seq) // 2:
        #     self.thread_safe_warn("Attempting to run BLAST job with a high occurrence of unknown sequences.")
        self.sequence_name = sequence_name  # string
        self.atomspec = atomspec  # string (atom specifier)
        self.database = database  # string
        self.cutoff = cutoff  # float
        self.matrix = matrix  # string
        self.max_seqs = max_seqs  # int
        if version is None:
            version = CurrentDBVersions[self.database]
        self.version = version  # DB Version
        self.log = log
        self.tool_inst_name = tool_inst_name

    def start(self):
        try:
            super().start(self.service_name, self.params)
        except MaxRetryError:
            self.session.logger.warning(self.inet_error)

    def _seq_to_fasta(self, seq, title):
        data = ["> %s\n" % title]
        block_size = 60
        for i in range(0, len(seq), block_size):
            data.append("%s\n" % seq[i : i + block_size])
        return "".join(data)

    def _params(self):
        return BlastParams(
            self.atomspec,
            self.database,
            self.cutoff,
            self.max_seqs,
            self.matrix,
            self.version,
        )

    def on_finish(self):
        logger = self.session.logger
        logger.status("BlastProtein finished.")
        if self.session.ui.is_gui:
            if self.exited_normally():
                if self.show_gui:
                    from .ui import BlastProteinResults

                    BlastProteinResults.from_job(
                        session=self.session,
                        tool_name=self.tool_inst_name,
                        params=self._params(),
                        job=self,
                        only_best=self.only_best,
                    )

                else:
                    params = self._params()
                    db = AvailableDBsDict[params.database]
                    hits, sequences = parse_blast_results(
                        params.database,
                        self.get_results(),
                        self.seq,
                        self.atomspec,
                    )
                    if self.only_best:
                        chains = defaultdict(str)
                        for hit in hits:
                            try:
                                chain, homotetramer = hit["name"].split("_")
                                if chain not in chains:
                                    chains[chain] = hit
                                else:
                                    old_homotetramer = chains[chain]["name"].split("_")[
                                        1
                                    ]
                                    best_homotetramer = sorted(
                                        [homotetramer, old_homotetramer]
                                    )[0]
                                    if best_homotetramer == homotetramer:
                                        chains[chain] = hit
                            except ValueError:
                                # If the chain doesn't have a homotetramer, just take the name
                                chain = hit["name"]
                                if chain not in chains:
                                    chains[chain] = hit
                                else:
                                    old_chain = chains[chain]["name"]
                                    best_chain = sorted([chain, old_chain])[0]
                                    if best_chain == chain:
                                        chains[chain] = hit
                        hits = list(chains.values())
                    if self.load_structures:
                        num_opened = 0
                        hits = sorted(hits, key=lambda i: i["e-value"])
                        _first_opened = None
                        for index, hit in enumerate(hits):
                            hit["hit_#"] = index + 1
                        for hit in hits:
                            name = hit[db.fetchable_col]
                            parts = name.split("_", 1)
                            if len(parts) == 1 or len(parts[0]) != 4:
                                continue
                            if params.database in ["alphafold", "esmfold"]:
                                models, chain_id = db.load_model(
                                    self.session, name, self.atomspec, params.version
                                )
                            else:
                                models, chain_id = db.load_model(
                                    self.session, name, self.atomspec
                                )
                            for m in models:
                                if self.atomspec:
                                    db.display_model(
                                        self.session,
                                        self.atomspec,
                                        m,
                                        chain_id,
                                    )
                                elif _first_opened:
                                    db.display_model(
                                        self.session, _first_opened, m, chain_id
                                    )
                                else:
                                    _first_opened = m.atomspec + "/" + chain_id
                                num_opened += 1
                        self.session.logger.info(
                            "Opened %s models, skipped %s sequence-only results"
                            % (num_opened, len(hits) - num_opened)
                        )
                    if self.load_sequences:
                        # Show the multiple alignment viewer
                        ids = [hit["id"] for hit in hits]
                        ids.insert(0, 0)
                        names = []
                        seqs = []
                        for sid in ids:
                            name, match = sequences[sid]
                            names.append(name)
                            seqs.append(match.sequence)
                        # Find columns that are gaps in all sequences and remove them.
                        all_gaps = set()
                        for i in range(len(seqs[0])):
                            for seq in seqs:
                                if seq[i].isalpha():
                                    break
                            else:
                                all_gaps.add(i)
                        if all_gaps:
                            for i in range(len(seqs)):
                                seq = seqs[i]
                                new_seq = "".join(
                                    [
                                        seq[n]
                                        for n in range(len(seq))
                                        if n not in all_gaps
                                    ]
                                )
                                seqs[i] = new_seq
                        # Generate multiple sequence alignment file
                        # Ask sequence viewer to display alignment
                        seqs = [
                            Sequence(name=name, characters=seqs[i])
                            for i, name in enumerate(names)
                        ]
                        name = "blast-nogui alignments"
                        # Ensure that the next time the user launches the same command that a
                        # unique index gets shown.
                        self.session.alignments.new_alignment(seqs, name)
            else:
                self.session.logger.error("BLAST job failed")
        else:
            if self.exited_normally():
                results = self.get_results()
                parse_blast_results_nogui(
                    self.session, self._params(), self.seq, results, self.log
                )
            else:
                self.session.logger.error("BLAST job failed")

    def __str__(self):
        return "BlastProtein Job, ID %s" % self.id

    @classmethod
    def from_snapshot(cls, session, data):
        params = data["params"]
        atomspec = data["atomspec"]
        seq = params["input_seq"]
        database = params["db"]
        cutoff = params["evalue"]
        matrix = params["matrix"]
        maxSeqs = params["blimit"]
        version = params["version"]
        tmp = cls(
            session,
            seq,
            atomspec,
            database=database,
            cutoff=cutoff,
            matrix=matrix,
            max_seqs=maxSeqs,
            version=version,
            log=None,
            tool_inst_name=data.get("tool_inst_name", None),
        )
        tmp.start_time = data["start_time"]
        tmp.end_time = data["end_time"]
        tmp.id = data["id"]
        tmp.job_id = data["job_id"]
        tmp.state = data["state"]
        tmp.restore()
        return tmp

    def take_snapshot(self, session, flags) -> Dict:
        data = super().take_snapshot(session, flags)
        data["params"] = self.params
        data["atomspec"] = self.atomspec
        data["tool_inst_name"] = self.tool_inst_name
        return data

    @staticmethod
    def restore_snapshot(session, data) -> "CxServicesJob":
        return BlastProteinJob.from_snapshot(session, data)


def parse_blast_results_nogui(session, params, sequence, results, log=None):
    blast_results = get_database(params.database)
    try:
        session.logger.info("Parsing BLAST results.")
        blast_results.parse("query", sequence, results)
    except Exception as e:
        session.logger.bug("BLAST output parsing error: %s" % str(e))
    else:
        if log or (log is None and not session.ui.is_gui):
            msgs = ["BLAST results for:"]
            for name, value in params._asdict().items():
                msgs.append("  %s: %s" % (name, value))
            for m in blast_results.parser.matches:
                name = m.match if m.match else m.name
                msgs.append(
                    "\t".join([name, "%.1e" % m.evalue, str(m.score), m.description])
                )
            session.logger.info("\n".join(msgs))
