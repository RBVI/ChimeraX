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
def alphafold_sequence_search(sequences, min_length=20, local=False, log=None):
    '''
    Search all AlphaFold database sequences using blat.
    Return best match uniprot ids.
    '''
    useqs = list(set(seq for seq in sequences if len(seq) >= min_length))
    if len(useqs) == 0:
        return [None] * len(sequences)

    if log is not None:
        log.status('Searching AlphaFold database for %d sequence%s'
                   % (len(useqs), _plural(useqs)))
                          
    if local:
        seq_uniprot_ids = _search_sequences_local(useqs)
    else:
        seq_uniprot_ids = _search_sequences_web(useqs)
    seq_uids = [seq_uniprot_ids.get(seq) for seq in sequences]
    
    return seq_uids

def _plural(seq):
    return 's' if len(seq) > 1 else ''

seq_database = '/Users/goddard/ucsf/data/alphafold/sequences/ref_proteomes/alphafold.fasta'
blat_exe = '/Users/goddard/ucsf/blat/bin/blat'
def _search_sequences_local(sequences, database_path=seq_database, blat_exe=blat_exe):

    # Make temporary directory for blat input and output files.
    from tempfile import TemporaryDirectory
    d = TemporaryDirectory(prefix = 'AlphaFold_Blat')
    dir = d.name
    from os.path import join
    query_path = join(dir, 'query.fasta')
    blat_output = join(dir, 'blat.out')
    
    # Write FASTA query file
    fq = open(query_path, 'w')
    fq.write(_fasta(sequences))
    fq.close()

    # Run BLAT
    args = (blat_exe, database_path, query_path, blat_output, '-out=blast8', '-prot')
    from subprocess import run
    status = run(args)
    if status.returncode != 0:
        raise RuntimeError('blat failed: %s' % ' '.join(args))

    # Parse blat results
    seq_uids = _parse_blat_output(blat_output, sequences)

    return seq_uids

def _parse_blat_output(blat_output, sequences):

    f = open(blat_output, 'r')
    out_lines = f.readlines()
    f.close()
    seq_uids = {}
    for line in out_lines:
        fields = line.split()
        s = int(fields[0])
        seq = sequences[s]
        if seq not in seq_uids:
            uniprot_id, uniprot_name = fields[1].split('|')[1:3]
            qstart, qend, mstart, mend = [int(p) for p in fields[6:10]]
            useq = UniprotSequence(uniprot_id, uniprot_name,
                                   (mstart, mend), (qstart, qend))
            seq_uids[seq] = useq

    return seq_uids

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

def _fasta(sequence_strings, LINELEN=60):
    lines = []
    for s,seq in enumerate(sequence_strings):
        lines.append('>%d' % s)
        for i in range(0, len(seq), LINELEN):
            lines.append(seq[i:i+LINELEN])
        lines.append('')
    return '\n'.join(lines)

sequence_search_url = 'https://www.rbvi.ucsf.edu/chimerax/cgi-bin/alphafold_search_cgi.py'
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
                                      (u['dbseq start'], u['dbseq end']),
                                      (u['query start'], u['query end']))
                for seq, u in zip(sequences, results['sequences']) if u}
    return seq_uids
        
class SearchError(RuntimeError):
    pass
