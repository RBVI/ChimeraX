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
def alphafold_sequence_search(sequences, min_length=20, log=None):
    '''
    Search AlphaFold database sequences using kmer search web service.
    Return best match uniprot ids.
    '''
    useqs = list(set(seq for seq in sequences if len(seq) >= min_length))
    if len(useqs) == 0:
        return [None] * len(sequences)

    if log is not None:
        log.status('Searching AlphaFold database for %d sequence%s'
                   % (len(useqs), _plural(useqs)))

    seq_uniprot_ids = _search_sequences_web(useqs)
    seq_uids = [seq_uniprot_ids.get(seq) for seq in sequences]

    return seq_uids

def _plural(seq):
    return 's' if len(seq) > 1 else ''

class UniprotSequence:
    def __init__(self, uniprot_id, uniprot_name,
                 database_sequence_range, query_sequence_range,
                 alphafold_database_version = '2'):
        self.uniprot_id = uniprot_id
        self.uniprot_name = uniprot_name
        self.database_sequence_range = database_sequence_range
        self.query_sequence_range = query_sequence_range
        self.range_from_sequence_match = True
        self.alphafold_database_version = alphafold_database_version

    def copy(self):
        return UniprotSequence(self.uniprot_id, self.uniprot_name,
                               self.database_sequence_range, self.query_sequence_range)

sequence_search_url = 'https://www.rbvi.ucsf.edu/chimerax/cgi-bin/alphafold_search3_cgi.py'
def _search_sequences_web(sequences, url = sequence_search_url):
    import json
    request = json.dumps({'sequences': sequences})
    import requests
    try:
        r = requests.post(url, data=request)
    except requests.exceptions.ConnectionError:
        raise SearchError('Unable to reach AlphaFold sequence search web service\n\n%s' % url)

    if r.status_code != 200:
        raise SearchError('AlphaFold sequence search web service failed (%s) "%s"\n\n%s'
                          % (r.status_code, r.reason, url))

    results = r.json()
    if 'error' in results:
        raise SearchError('AlphaFold sequence search web service\n\n%s\n\nreported error:\n\n%s'
                          % (url, results['error']))
    seq_uids = {seq : UniprotSequence(u['uniprot id'], u['uniprot name'],
                                      (u.get('dbseq start'), u.get('dbseq end')),
                                      (u.get('query start'), u.get('query end')),
                                      u['db version'])
                for seq, u in zip(sequences, results['sequences']) if u}
    return seq_uids

class SearchError(RuntimeError):
    pass
