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

# -----------------------------------------------------------------------------
# Search ESMFold database for sequences
#
def esmfold_sequence_search(sequences, min_length=20, log=None):
    '''
    Search all ESMFold database sequences using kmer-search.
    Return best match mgnify ids.
    '''
    useqs = list(set(seq for seq in sequences if len(seq) >= min_length))
    if len(useqs) == 0:
        return [None] * len(sequences)

    if log is not None:
        log.status('Searching ESM Metagenomic Atlas for %d sequence%s'
                   % (len(useqs), _plural(useqs)))

    seq_mgnify_ids = _search_sequences_web(useqs)
    seq_mgids = [seq_mgnify_ids.get(seq) for seq in sequences]

    return seq_mgids

def _plural(seq):
    n = seq if isinstance(seq, int) else len(seq)
    return 's' if n > 1 else ''

from chimerax.alphafold.search import DatabaseEntryId

sequence_search_url = 'https://www.rbvi.ucsf.edu/chimerax/cgi-bin/esmfold_search_cgi.py'
def _search_sequences_web(sequences, url = sequence_search_url):
    import json
    request = json.dumps({'sequences': sequences})
    import requests
    try:
        r = requests.post(url, data=request)
    except requests.exceptions.ConnectionError:
        raise SearchError('Unable to reach ESMFold sequence search web service\n\n%s' % url)

    if r.status_code != 200:
        raise SearchError('ESMFold sequence search web service failed (%s) "%s"\n\n%s'
                          % (r.status_code, r.reason, url))

    results = r.json()
    if 'error' in results:
        raise SearchError('ESMFold sequence search web service\n\n%s\n\nreported error:\n\n%s'
                          % (url, results['error']))

    seq_uids = {seq : DatabaseEntryId(hit['mgnify id'], id_type = 'MGnify',
                                      database = 'ESMFold', version = hit['db version'])
                for seq, hit in zip(sequences, results['sequences']) if hit}
    return seq_uids

class SearchError(RuntimeError):
    pass
