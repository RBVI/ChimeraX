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
def chain_sequence_search(chains, min_length=20, local=False):
    '''
    Search all AlphaFold database sequences using blat.
    Return best match uniprot ids.
    '''
    sequences = list(set(chain.characters for chain in chains
                         if len(chain.characters) >= min_length))
    if len(sequences) == 0:
        return {}

    session = chains[0].structure.session
    session.logger.status('Searching AlphaFold database for %d sequences' % len(sequences))
                          
    if local:
        seq_uniprot_ids = _search_sequences_local(sequences)
    else:
        seq_uniprot_ids = _search_sequences_web(sequences)
    chain_uids = [(chain, seq_uniprot_ids[chain.characters].copy(chain.chain_id))
                  for chain in chains if chain.characters in seq_uniprot_ids]
    
    return chain_uids

seq_database = '/Users/goddard/ucsf/chimerax/src/bundles/alphafold/src/sequences/ref_proteomes/alphafold.fasta'
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
            useq = UniprotSequence(None, uniprot_id, uniprot_name,
                                   (mstart, mend), (qstart, qend))
            seq_uids[seq] = useq

    return seq_uids

class UniprotSequence:
    def __init__(self, chain_id, uniprot_id, uniprot_name,
                 database_sequence_range, chain_sequence_range):
        self.chain_id = chain_id
        self.uniprot_id = uniprot_id        
        self.uniprot_name = uniprot_name
        self.database_sequence_range = database_sequence_range
        self.chain_sequence_range = chain_sequence_range
    def copy(self, chain_id):
        return UniprotSequence(chain_id, self.uniprot_id, self.uniprot_name,
                            self.database_sequence_range, self.chain_sequence_range)

def _fasta(sequence_strings, LINELEN=60):
    lines = []
    for s,seq in enumerate(sequence_strings):
        lines.append('>%d' % s)
        for i in range(0, len(seq), LINELEN):
            lines.append(seq[i:i+LINELEN])
        lines.append('')
    return '\n'.join(lines)

#sequence_search_url = 'http://localhost/cgi-bin/alphafold_search_cgi.py'
#sequence_search_url = 'https://preview.rbvi.ucsf.edu/chimerax/cgi-bin/alphafold_search_cgi.py'
sequence_search_url = 'https://www.rbvi.ucsf.edu/chimerax/cgi-bin/alphafold_search_cgi.py'
def _search_sequences_web(sequences, url = sequence_search_url):
    import json
    request = json.dumps({'sequences': sequences})
    import requests
    r = requests.post(url, data=request)
    results = r.json()
    if 'error' in results:
        raise RuntimeError('Web service reported error:\n%s' % results['error'])
    if r.status_code != 200:
        raise RuntimeError('Web service failed, return code %s' % r.status_code)
    seq_uids = {seq : UniprotSequence(None, u['uniprot id'], u['uniprot name'],
                                      (u['dbseq start'], u['dbseq end']),
                                      (u['query start'], u['query end']))
                for seq, u in zip(sequences, results['sequences']) if u}
    return seq_uids
        
