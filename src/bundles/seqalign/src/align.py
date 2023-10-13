# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
import importlib
import io
import tempfile

from chimerax.core.errors import NonChimeraXError
from .cmd import MUSCLE, CLUSTAL_OMEGA

# Implement only as blocking for now; can add non-blocking later if any need for it.
# Also, no options for now, can add later if any demand.
def realign_sequences(session, sequences, *, program=CLUSTAL_OMEGA):
    realigned_sequences = []
    # create temp file in main line of function, so that it is not closed/garbage collected prematurely
    input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False)
    from chimerax.atomic import Sequence
    class FakeAlignment:
        seqs = [Sequence(name=s.name, characters=s.ungapped()) for s in sequences]
    mod = importlib.import_module(".io.saveFASTA", "chimerax.seqalign")
    mod.save(session, FakeAlignment(), input_file)
    input_file.close()
    """
    from chimerax.webservices.opal_job import OpalJob
    class RealignJob(OpalJob):
        def __init__(self):
            super().__init__(session)
            service_name, options, self.reorders_seqs, in_flag, out_flag = {
                MUSCLE: (
                    "MuscleService",
                    "-maxiters 1",
                    True,
                    "-in",
                    "-out",
                ),
                CLUSTAL_OMEGA: (
                    "ClustalOmegaService",
                    "--iterations 1 --full --full-iter",
                    False,
                    "-i",
                    "-o",
                ),
            }[program]
            command = "%s input.fa %s output.fa %s" % (in_flag, out_flag, options)
            session.logger.status("Starting %s alignment" % program)
            input_file_map = [ ("input.fa", "text_file", input_file.name) ]
            self.start(service_name, command, input_file_map=input_file_map, blocking=True)
    """
    from chimerax.webservices.cxservices_job import CxServicesJob
    class RealignJob(CxServicesJob):
        def __init__(self):
            super().__init__(session)
            service_name, options, self.reorders_seqs, in_flag, out_flag = {
                MUSCLE: (
                    "muscle",
                    {'maxiters': "1"},
                    True,
                    "-in",
                    "-out",
                ),
                CLUSTAL_OMEGA: (
                    "clustal_omega",
                    {"iterations": "1"}, #  "--full", "--full-iter",
                    False,
                    "-i",
                    "-o",
                ),
            }[program]
            #command = "%s input.fa %s output.fa %s" % (in_flag, out_flag, options)
            params = {
                'in_flag': in_flag,
                'out_flag': out_flag
            }
            params.update(options)
            session.logger.status("Starting %s alignment" % program)
            #input_file_map = [ ("input.fa", "text_file", input_file.name) ]
            self.start(service_name, params, [input_file.name], blocking=True)

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
    job = RealignJob()
    return realigned_sequences
