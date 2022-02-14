# vim: set expandtab ts=4 sw=4:

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
import importlib
import io
import tempfile

from chimerax.core.errors import NonChimeraXError
from chimerax.webservices import CxServicesJob
from .cmd import MUSCLE, CLUSTAL_OMEGA

# Implement only as blocking for now; can add non-blocking later if any need for it.
# Also, no options for now, can add later if any demand.
def realign_sequences(session, sequences, *, program=CLUSTAL_OMEGA):
    realigned_sequences = []
    # create temp file in main line of function, so that it is not closed/garbage collected prematurely
    input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False)
    class FakeAlignment:
        seqs = sequences
    mod = importlib.import_module(".io.saveFASTA", "chimerax.seqalign")
    mod.save(session, FakeAlignment(), input_file)
    input_file.close()

    class RealignCxServicesJob(CxServicesJob):
        def __init__(self, program, input_file):
            super().__init__(session)
            self.service_name = program.lower().replace(' ','_')
            self.params = None
            self.reorder_seqs = None
            if self.service_name == 'muscle':
                self.params = {
                    "maxiters": 1,

                }
                self.reorder_seqs = True
            elif self.service_name == 'clustal_omega':
                self.params = {
                    "iters": 1,
                    "full": True,
                    "full_iter": True,
                }
                self.reorder_seqs = False
            session.logger.status("Starting %s alignment" % program)
            self.input_file = [input_file]
            self.start(self.service_name, self.params, self.input_file)

        def on_finish(self):
            logger = session.logger
            logger.status("%s alignment finished" % program)
            if not self.exited_normally():
                err = self.get_file("stderr.txt")
                if err:
                    raise RuntimeError(("%s failure; standard error:\n" % program) + err)
                else:
                    raise RuntimeError("%s failure with no error output" % program)
            try:
                fasta_output = self.get_file("output.fa")
            except KeyError:
                try:
                    stdout = self.get_file("stdout.txt")
                    stderr = self.get_file("stderr.txt")
                except KeyError:
                    raise RuntimeError("No output from %s" % program)
                logger.info("<br><b>%s error output</b>" % program, is_html=True)
                logger.info(stderr)
                logger.info("<br><b>%s run output</b>" % program, is_html=True)
                logger.info(stdout)
                raise NonChimeraXError("No output alignment from %s; see log for %s text output"
                    % (program, program))
            mod = importlib.import_module(".io.readFASTA", "chimerax.seqalign")
            out_seqs, *args = mod.read(session, io.StringIO(fasta_output))
            if self.reorders_seqs:
                # put result in the same order as the original sequences
                orig_names = set([s.name for s in sequences])
                new_names = set([s.name for s in out_seqs])
                if orig_names == new_names:
                    # names match, so correct reordering is possible
                    order = {}
                    if len(new_names) == len(sequences):
                        # names are unique, so use simpler sorting
                        for i, s in enumerate(sequences):
                            order[s.name] = i
                        key_func = lambda s: order[s.name]
                    else:
                        for i, s in enumerate(sequences):
                            order[(s.name, s.ungapped())] = i
                        key_func = lambda s: order[(s.name, s.ungapped())]
                    out_seqs.sort(key=key_func)
            realigned_sequences.extend(out_seqs)

    job = RealignCxServicesJob(program, input_file.name)
    return realigned_sequences
