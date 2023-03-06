# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from urllib3.exceptions import MaxRetryError

from chimerax.core.tasks import JobError
from chimerax.webservices.cxservices_job import CxServicesJob

from .data_model import get_database, CurrentDBVersions
from .utils import BlastParams, make_instance_name

class BlastProteinJob(CxServicesJob):
    inet_error = "Could not start BLAST job. Please check your internet connection and try again."
    service_name = "blast"

    def __init__(self, session, seq, atomspec, **kw):
        super().__init__(session)

        if 'tool_inst_name' not in kw:
            kw['tool_inst_name'] = make_instance_name()
        if kw['tool_inst_name'] is None:
            kw['tool_inst_name'] = make_instance_name()

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
            "version": self.version
        }

        try:
            model_no = int(atomspec.split('/')[0].split('#')[1])
            self.model_name = session.models._models[(model_no,)]._name
        except (ValueError, KeyError, AttributeError):
            self.model_name = None

        try:
            self.start(self.service_name, self.params)
        except MaxRetryError:
            session.logger.warning(self.inet_error)

    def setup(self, seq, atomspec, database: str = "pdb", cutoff: float = 1.0e-3,
              matrix: str = "BLOSUM62", max_seqs: int = 100, log = None,
              version = None, tool_inst_name = None, sequence_name = None):
        self.seq = seq.replace('?', 'X')                  # string
        if self.seq.count('X') == len(self.seq):
            raise JobError("Sequence consists entirely of unknown amino acids.")
        # if self.seq.count('X') > len(self.seq) // 2:
        #     self.thread_safe_warn("Attempting to run BLAST job with a high occurrence of unknown sequences.")
        self.sequence_name = sequence_name                # string
        self.atomspec = atomspec                          # string (atom specifier)
        self.database = database                          # string
        self.cutoff = cutoff                              # float
        self.matrix = matrix                              # string
        self.max_seqs = max_seqs                          # int
        if version is None:
            version = CurrentDBVersions[self.database]
        self.version = version                            # DB Version
        self.log = log
        self.tool_inst_name = tool_inst_name

    def _seq_to_fasta(self, seq, title):
        data = ["> %s\n" % title]
        block_size = 60
        for i in range(0, len(seq), block_size):
            data.append("%s\n" % seq[i:i + block_size])
        return ''.join(data)

    def _params(self):
        return BlastParams(
            self.atomspec, self.database, self.cutoff
            , self.max_seqs, self.matrix, self.version
        )

    def on_finish(self):
        logger = self.session.logger
        logger.status("BlastProtein finished.")
        if self.session.ui.is_gui:
            from .ui import BlastProteinResults
            BlastProteinResults.from_job(
                    session = self.session
                    , tool_name = self.tool_inst_name
                    , params=self._params()
                    , job=self
            )
        else:
            if self.exited_normally():
                results = self.get_results()
                parse_blast_results_nogui(self.session, self._params(), self.seq, results, self.log)
            else:
                self.session.logger.error("BLAST job failed")

    def __str__(self):
        return "BlastProtein Job, ID %s" % self.id


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
                msgs.append('\t'.join([name, "%.1e" % m.evalue,
                                       str(m.score),
                                       m.description]))
            session.logger.info('\n'.join(msgs))
