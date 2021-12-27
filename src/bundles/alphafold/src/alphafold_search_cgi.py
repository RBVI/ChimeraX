#!/usr/local/bin/python3

# CGI script to read list of sequences and search an AlphaFold sequence
# database using blat.

# On plato.cgl.ucsf.edu:
database_path = '/databases/mol/AlphaFold/v1/alphafold.fasta'
blat_exe = '/usr/local/bin/blat'

def parse_request_sequences():
    import os
    content_length = int(os.environ['CONTENT_LENGTH'])

    import sys
    request_body = sys.stdin.read(content_length)

    import json
    query_sequences = json.loads(request_body)

    return query_sequences['sequences']

def report_matches(matches):
    import json
    lines = [
        'Content-Type: application/json',
        '',
        json.dumps(matches)
    ]
    msg = '\n'.join(lines)
    import sys
    sys.stdout.write(msg)

def search_sequences(sequences, database_path=database_path, blat_exe=blat_exe):

    # Make temporary directory for blat input and output files.
    from tempfile import TemporaryDirectory
    d = TemporaryDirectory(prefix = 'AlphaFold_Blat')
    dir = d.name
    from os.path import join
    query_path = join(dir, 'query.fasta')
    blat_output = join(dir, 'blat.out')
    
    # Write FASTA query file
    fq = open(query_path, 'w')
    fq.write(fasta(sequences))
    fq.close()

    # Run BLAT
    args = (blat_exe, database_path, query_path, blat_output, '-out=blast8', '-prot')
    from subprocess import run, DEVNULL
    status = run(args, stdout=DEVNULL, stderr=DEVNULL)
    if status.returncode != 0:
        raise RuntimeError('blat failed: %s' % ' '.join(args))

    # Parse blat results
    seq_uids = parse_blat_output(blat_output, sequences)

    matches = [seq_uids.get(sequence, {}) for sequence in sequences]
    return matches

def parse_blat_output(blat_output, sequences):

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
            seq_uids[seq] = {'uniprot id': uniprot_id,
                             'uniprot name': uniprot_name,
                             'dbseq start': mstart,
                             'dbseq end': mend,
                             'query start': qstart,
                             'query end': qend,
            }

    return seq_uids

def fasta(sequence_strings, LINELEN=60):
    lines = []
    for s,seq in enumerate(sequence_strings):
        lines.append('>%d' % s)
        for i in range(0, len(seq), LINELEN):
            lines.append(seq[i:i+LINELEN])
        lines.append('')
    return '\n'.join(lines)

def search_database():
    try:
        sequences = parse_request_sequences()
        matches = search_sequences(sequences)
        report_matches({'sequences':matches})
    except Exception:
        import sys, json, traceback
        sys.stdout.write('Content-Type: application/json\n\n%s'
                         % json.dumps({'error': traceback.format_exc()}))

if __name__ == '__main__':
    search_database()

    
