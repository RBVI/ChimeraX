# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.webservices.opal_job import OpalJob


class CCDJob(OpalJob):

    OPAL_SERVICE = "CCDService"

    def __init__(self, session, name):
        super().__init__(session)
        self.start(self.OPAL_SERVICE, name)

    def on_finish(self):
        self.session.logger.info("Standard output:\n" +
                                 self.get_file("stdout.txt"))


class BlastProteinJob(OpalJob):

    OPAL_SERVICE = "BlastProtein2Service"
    QUERY_FILENAME = "query.fa"
    RESULTS_FILENAME = "results.txt"

    def __init__(self, session, seq, atomspec, database="pdb", cutoff=1.0e-3,
                 matrix="BLOSUM62", max_hits=500, log=None, tool=None):
        super().__init__(session)
        self.seq = seq                          # string
        self.atomspec = atomspec                # string (atom specifier)
        self.database = database                # string
        self.cutoff = cutoff                    # float
        self.matrix = matrix                    # string
        self.max_hits = max_hits                # int
        self.log = log
        self.tool = tool

        options = ["-d", self.database,
                   "-e", str(self.cutoff),
                   "-M", self.matrix,
                   "-b", str(self.max_hits),
                   "-i", self.QUERY_FILENAME,
                   "-o", self.RESULTS_FILENAME]
        cmd = ' '.join(options)
        fasta = self._seq_to_fasta(self.seq, "query")
        input_file_map = [(self.QUERY_FILENAME, "bytes", fasta.encode("utf-8"))]
        self.start(self.OPAL_SERVICE, cmd, input_file_map=input_file_map)

    def _seq_to_fasta(self, seq, title):
        data = ["> %s\n" % title]
        block_size = 60
        for i in range(0, len(seq), block_size):
            data.append("%s\n" % seq[i:i+block_size])
        return ''.join(data)

    def on_finish(self):
        logger = self.session.logger
        logger.info("BlastProtein finished.")
        out = self.get_file("stdout.txt")
        if out:
            logger.error("Standard output:\n" + out)
        if not self.exited_normally():
            err = self.get_file("stderr.txt")
            if self.tool:
                self.tool.job_failed(self, err)
            else:
                if err:
                    logger.bug("Standard error:\n" + err)
        else:
            from .blastp_parser import Parser
            results = self.get_file(self.RESULTS_FILENAME)
            try:
                p = Parser("query", self.seq, results)
            except ValueError as e:
                if self.tool:
                    self.tool.job_failed(self, str(e))
                else:
                    logger.bug("BLAST output parsing error: %s" % str(e))
            else:
                if self.tool:
                    self.tool.job_finished(self, p)
                else:
                    if self.session.ui.is_gui:
                        from .tool import ToolUI
                        ToolUI(self.session, "BlastProtein",
                               blast_results=p, atomspec=self.atomspec)
                    if self.log or (self.log is None and
                                    not self.session.ui.is_gui):
                        msgs = ["BLAST results:"]
                        for m in p.matches:
                            name = m.pdb if m.pdb else m.name
                            msgs.append('\t'.join([name, "%.1e" % m.evalue,
                                                   str(m.score),
                                                   m.description]))
                        logger.info('\n'.join(msgs))
