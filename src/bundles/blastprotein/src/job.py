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
from chimerax.webservices.cxservices_job import CxServicesJob
from cxservices.rest import ApiException

from . import tool

from .datatypes import BlastParams
from .databases import Database, get_database
from .results import BlastProteinResults
from .utils import make_instance_name

class BlastProteinJob(CxServicesJob):
    QUERY_FILENAME = "query.fa"
    RESULTS_FILENAME = "results.json"

    def __init__(self, session, seq, atomspec, **kw):
        super().__init__(session)
        if 'tool_inst_name' not in kw:
            kw['tool_inst_name'] = make_instance_name()
        if kw['tool_inst_name'] is None:
            kw['tool_inst_name'] = make_instance_name()
        self.setup(seq, atomspec, **kw)
        params = {"db": self.database,
                  "evalue": str(self.cutoff),
                  "matrix": self.matrix,
                  "blimit": str(self.max_seqs),
                  "input_seq": self.seq,
                  "output_file": self.RESULTS_FILENAME}
        try:
            self.start("blast", params)
        except MaxRetryError as e:
            session.logger.warning("Could not start BLAST job. Please check your internet connection and try again.")

    def setup(self, seq, atomspec, database: str ="pdb", cutoff: float = 1.0e-3,
              matrix: str="BLOSUM62", max_seqs: int=500, log=None, tool_inst_name=None,
              sequence_name=None):
        self.seq = seq.replace('?', 'X')                  # string
        self.sequence_name = sequence_name                # string
        self.atomspec = atomspec                          # string (atom specifier)
        self.database = database                          # string
        self._database: Database = get_database(database) # object
        self.cutoff = cutoff                              # float
        self.matrix = matrix                              # string
        self.max_seqs = max_seqs                          # int
        self.log = log
        self.tool_inst_name = tool_inst_name

    def _seq_to_fasta(self, seq, title):
        data = ["> %s\n" % title]
        block_size = 60
        for i in range(0, len(seq), block_size):
            data.append("%s\n" % seq[i:i+block_size])
        return ''.join(data)

    def _params(self):
        return BlastParams(self.atomspec, self.database, self.cutoff, self.max_seqs, self.matrix)

    def on_finish(self):
        logger = self.session.logger
        logger.info("BlastProtein finished.")
        if self.session.ui.is_gui:
            BlastProteinResults(session = self.session
                                , tool_name = self.tool_inst_name
                                , job=self
                                , params=self._params())
        else:
            out = self.get_stdout()
            if out:
                logger.error("Standard output:\n" + out)
            if not self.exited_normally():
                err = self.get_stderr()
                if err:
                    logger.bug("Standard error:\n" + err)
            else:
                results = self.get_file(self.RESULTS_FILENAME)
                try:
                    logger.info("Parsing BLAST results.")
                    self._database.parse("query", self.seq, results)
                except Exception as e:
                    logger.bug("BLAST output parsing error: %s" % str(e))
                else:
                    if self.log or (self.log is None and not self.session.ui.is_gui):
                        msgs = ["BLAST results for:"]
                        for name, value in self._params()._asdict():
                            msgs.append("  %s: %s" % (name, value))
                        for m in self._database.parser.matches:
                            name = m.match if m.match else m.name
                            msgs.append('\t'.join([name, "%.1e" % m.evalue,
                                                   str(m.score),
                                                   m.description]))
                        logger.info('\n'.join(msgs))
