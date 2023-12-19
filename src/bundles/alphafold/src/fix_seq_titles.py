# Merge uniprot sequence fasta title lines with AlphaFold sequence fasta file
# to get species and uniprot names for alphafold sequences.
def find_deleted_entries():
    suids = set()
    f = open('sequences.fasta', 'r')
    for line in f.readlines():
        if line.startswith('>'):
            uid = line.split('-')[1]
            suids.add(uid)
    f.close()
    print ('unique sequences.fasta', len(suids), list(suids)[:5])

    uuids = set()
    f = open('uniprot_fetch/uniprot_seqs.fasta', 'r')
    for line in f.readlines():
        if line.startswith('>'):
            uid = line.split('|')[1]
            uuids.add(uid)
    f.close()
    print ('unique uniprot_seqs.fasta', len(uuids), list(uuids)[:5])

    muids = suids - uuids
    print ('sequences.fasta - uniprot_seq.fasta', len(muids), list(muids)[:10])

def replace_titles():
    titles = {}
    f = open('uniprot_fetch/uniprot_seqs.fasta', 'r')
    for line in f.readlines():
        if line.startswith('>'):
            uid = line.split('|')[1]
            titles[uid] = line
    f.close()

    import sys
    f = open('sequences.fasta', 'r')
    for line in f.readlines():
        if line.startswith('>'):
            uid = line.split('-')[1]
            if uid in titles:
                title = titles[uid]
            else:
                title = f'>xx|{uid}|deleted deleted OS=deleted\n'
            sys.stdout.write(title)
        else:
            sys.stdout.write(line)
    f.close()

replace_titles()

    
