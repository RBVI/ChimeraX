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
    n = seq if isinstance(seq, int) else len(seq)
    return 's' if n > 1 else ''

from chimerax.core.state import State
class DatabaseEntryId(State):
    '''Database entry identifier for a predicted structure.'''
    def __init__(self, id, name = None, id_type = 'UniProt', database = 'AlphaFold', version = None):
        self.id = id			# 'Q6JC40'
        self.id_type = id_type		# 'UniProt'
        self.name = name		# 'Q6JC40_9VIRU'
        self.database = database	# 'AlphaFold'
        self.version = version		# '3'

    # State save/restore in ChimeraX
    _save_attrs = ['id', 'id_type', 'name', 'database', 'version']
    def take_snapshot(self, session, flags):
        data = {attr: getattr(self, attr) for attr in self._save_attrs}
        return data
    @staticmethod
    def restore_snapshot(session, data):
        return DatabaseEntryId(**data)
    
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
    seq_uids = {seq : DatabaseEntryId(u['uniprot id'], name = u['uniprot name'], version = u['db version'])
                for seq, u in zip(sequences, results['sequences']) if u}
    return seq_uids

class SearchError(RuntimeError):
    pass
