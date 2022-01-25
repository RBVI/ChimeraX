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
import json

from urllib3.exceptions import MaxRetryError
from chimerax.core.tasks import JobError
from chimerax.webservices.cxservices_job import CxServicesJob
from chimerax.webservices.cxservices_utils import (
    get_status, get_stdout, get_stderr, get_file
)
from cxservices.rest import ApiException

from .data_model import get_database
from .ui import BlastProteinResults
from .utils import BlastParams, make_instance_name

class BlastProteinJob(CxServicesJob):
    QUERY_FILENAME = "query.json"
    RESULTS_FILENAME = "results.json"

    inet_error = "Could not start BLAST job. Please check your internet connection and try again."

    def __init__(self, session, seq, atomspec, **kw):
        super().__init__(session)
        if 'tool_inst_name' not in kw:
            kw['tool_inst_name'] = make_instance_name()
        if kw['tool_inst_name'] is None:
            kw['tool_inst_name'] = make_instance_name()
        self.setup(seq, atomspec, **kw)
        self.params = {
            "db": self.database,
            "evalue": str(self.cutoff),
            "matrix": self.matrix,
            "blimit": str(self.max_seqs),
            "input_seq": self.seq,
            "output_file": self.RESULTS_FILENAME,
            "version": self.version
        }
        try:
            self.start("blast", self.params)
        except MaxRetryError:
            session.logger.warning(self.inet_error)

    def setup(self, seq, atomspec, database: str = "pdb", cutoff: float = 1.0e-3,
              matrix: str = "BLOSUM62", max_seqs: int = 100, log = None, tool_inst_name = None,
              sequence_name = None):
        self.seq = seq.replace('?', 'X')                  # string
        self.sequence_name = sequence_name                # string
        self.atomspec = atomspec                          # string (atom specifier)
        self.database = database                          # string
        self.cutoff = cutoff                              # float
        self.matrix = matrix                              # string
        self.max_seqs = max_seqs                          # int
        self.version = "2"                                # AlphaFold DB Version
        self.log = log
        self.tool_inst_name = tool_inst_name

    def _seq_to_fasta(self, seq, title):
        data = ["> %s\n" % title]
        block_size = 60
        for i in range(0, len(seq), block_size):
            data.append("%s\n" % seq[i:i + block_size])
        return ''.join(data)

    def _params(self):
        return BlastParams(self.atomspec, self.database, self.cutoff, self.max_seqs, self.matrix, self.version)

    def on_finish(self):
        logger = self.session.logger
        logger.info("BlastProtein finished.")
        if self.session.ui.is_gui:
            BlastProteinResults.from_job(
                    session = self.session
                    , tool_name = self.tool_inst_name
                    , params=self._params()
                    , job=self
            )
        else:
            out = err = results = None
            out = self.get_stdout()
            if out:
                logger.error("Standard output:\n" + out)
            if not self.exited_normally():
                err = self.get_stderr()
                if err:
                    logger.bug("Standard error:\n" + err)
            else:
                results = self.get_file(self.RESULTS_FILENAME)
                parse_blast_results_nogui(self.session, self._params(), self.seq, results, self.log)

    def __str__(self):
        return "BlastProtein Job, ID %s" % self.id


def manually_pull_blast_job(session, job_id, log=None):
    # TODO: Fetch the protein on which the BLAST was run from the server and open it
    # before displaying the BLAST results.
    """
    Pull a previously completed job, if it still exists on the remote server.

    Parameters:
        job_id: The job ID returned by the remote server.
        encoding: The remote file's expected encoding. By default, UTF-8.
    """
    try:
        status = get_status(job_id)
    except ApiException:
        session.logger.warning("Could not fetch job status; please double-check job ID and try again.")
        return
    else:
        err_str = "Could not pull results:"
        if status == "failed":
            session.logger.warning(" ".join([err_str, "job failed"]))
            return
        elif status == "deleted":
            session.logger.warning(" ".join([err_str, "job deleted from server"]))
            return
        elif status == "pending":
            session.logger.warning(" ".join([err_str, "job pending"]))
            return
        elif status == "running":
            session.logger.warning(" ".join([err_str, "job is still running"]))
            return
    try:
        stdout = get_stdout(job_id)
        stderr = get_stderr(job_id)
        if stdout:
            raise JobError(stdout)
        if stderr:
            raise JobError(stderr)
    except JobError as e:
        session.logger.bug("Job reported an output stream:\n" + str(e))
        return
    except ApiException:
        session.logger.bug("Job found but could not fetch stdout or stdin.")
        return
    raw_results = get_file(job_id, BlastProteinJob.RESULTS_FILENAME)
    params = get_file(job_id, BlastProteinJob.QUERY_FILENAME)
    params = BlastParams(**json.loads(params))
    sequence = get_file(job_id, '_stdin').split('\n')[1]
    job_id = job_id
    tool_inst_name = make_instance_name()
    if session.ui.is_gui:
        BlastProteinResults.from_pull(
            session = session
            , tool_name = tool_inst_name
            , params = params
            , sequence = sequence
            , results = raw_results
        )
    else:
        parse_blast_results_nogui(session, params, sequence, raw_results, log)

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
