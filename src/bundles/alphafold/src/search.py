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

# -----------------------------------------------------------------------------
# Search AlphaFold database for sequences
#

from chimerax.webservices.cxservices_job import CxServicesJob

def alphafold_sequence_search(sequences, minlength=20, local=False, session):
    '''
    Search all AlphaFold database sequences using blat.
    Return best match uniprot ids.
    '''
    useqs = list(set(seq for seq in sequences if len(seq) >= min_length))
    if len(useqs) == 0:
        return [None] * len(sequences)

    if session is not None:
        session.logger.log.status('Searching AlphaFold database for %d sequence%s'
                   % (len(useqs), _plural(useqs)))

    search_job = BlatJob(session, sequences, report_status_messages=False)

    # seq_uids = [seq_uniprot_ids.get(seq) for seq in sequences]
    try:
        seq_uids = search_job.on_finish
    except:
        session.logger.bug("Failed to create chain_uids.")
        return {}
    else:
        return seq_uids

def _plural(seq):
    return 's' if len(seq) > 1 else ''

class UniprotSequence:
    def __init__(self, uniprot_id, uniprot_name,
                 database_sequence_range, query_sequence_range):
        self.uniprot_id = uniprot_id        
        self.uniprot_name = uniprot_name
        self.database_sequence_range = database_sequence_range
        self.query_sequence_range = query_sequence_range
        self.range_from_sequence_match = True
    def copy(self):
        return UniprotSequence(self.uniprot_id, self.uniprot_name,
                               self.database_sequence_range, self.query_sequence_range)

class SearchError(RuntimeError):
    pass

class BlatJob(CxServicesJob):

    RESULTS_FILENAME = "blat.out"

    def __init__(self, session, sequences, report_status_messages = True):
        super().__init__(session)
        self.sequences = sequences
        self.report_status_messages = report_status_messages
        self.params = {
            "sequences": sequences,
            "output_file": self.RESULTS_FILENAME
        }
        self.start("blat", self.params, report_jobid = self.report_status_messages)

    def get_chain_ids(self, chains):
        await self.on_finish
        return [(chain, self.seq_uniprot_ids[chain.characters].copy(chain.chain_id))
                for chain in chains if chain.characters in self.seq_uniprot_ids]

    def parse_blat_output(self, blat_output):
        seq_uids = {}
        for line in blat_output.split('\n')[:-1]: # We don't want the final blank line
            fields = line.split()
            s = int(fields[0])
            seq = self.sequences[s]
            if seq not in seq_uids:
                uniprot_id, uniprot_name = fields[1].split('|')[1:]
                qstart, qend, mstart, mend = [int(float(p)) for p in fields[6:10]]
                useq = UniprotSequence(None, uniprot_id, uniprot_name,
                                       (mstart, mend), (qstart, qend))
                seq_uids[seq] = useq
        return seq_uids

    async def on_finish(self):
        # Modelled after BlastProteinJob.on_finish()
        logger = self.session.logger
        if self.report_status_messages:
            logger.info("AlphaFold BLAT finished.")
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
                return self.parse_blat_output(results)
            except Exception as e:
                logger.bug("BLAT output parsing error: %s" % str(e))
