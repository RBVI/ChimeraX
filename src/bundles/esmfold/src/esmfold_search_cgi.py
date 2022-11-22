#!/usr/local/bin/python3

# CGI script to read list of sequences and search an ESMFold sequence
# database using k-mer search.

# On plato.cgl.ucsf.edu:
database_version = '1'
database_path = f'/databases/mol/ESMFold/v{database_version}_kmer/esmfold.fasta'

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

def sequence_search(sequences, database_path = database_path, min_kmer_matches = 15,
                    max_sequence_length = 10000, nthreads = 16):
    '''
    Search for each of the specified sequences in a database of sequences
    given by the FASTA file at database_path.  The k-mer index must already
    have been created for the database file using the create_index_files() routine.
    For each sequence the uniprot accession code and name, the best matching sequence,
    and the number of matching k-mers is returned.  If there is no matching sequence
    with the minumum number of matching k-mers then None is returned for that sequence.
    '''
    results = []
    kmer_index = KmerSequenceIndex(database_path)
    for sequence in sequences:
        seq = clean_sequence(sequence)
        if len(seq) > max_sequence_length:
            results.append({})
            continue
        seq_num, num_kmer_matches = kmer_index.search(seq, nthreads = nthreads)
        if num_kmer_matches < min_kmer_matches:
            results.append({})
            continue
        title, db_sequence = kmer_index.title_and_sequence(seq_num)
        
        results.append(accession_codes(title, database_version))
    return results
    
class KmerSequenceIndex:
    '''
    Search for a sequence in a database of sequences given in a FASTA file.
    The FASTA file of sequences must have been processed to create index files
    using the create_index_files() routine to quickly lookup sequences containing a k-mer.
    A k-mer is a short sequence of k amino acids.
    '''
    
    def __init__(self, sequences_path, k = 5):
        self._sequences_path = sequences_path	# FASTA sequence file
        self.k = k
        self.counts = None	# Array containing number of sequences for each k-mer
        self.sizes = None	# Array containing number of bytes for each FASTA file entry.

    def search(self, sequence, nthreads = 1):
        '''
        Return the database sequence number which has the most k-mers matching
        the specified sequence.  Also return the number of matching k-mers.
        '''
        if self.counts is None:
            self.load_index()

        # List the k-mers in the given sequence.
        kmers = sequence_kmers(sequence, self.k)

        # Find the database sequences that have these k-mers.
        seqi_list = self.kmer_sequences(kmers, nthreads=nthreads)

        # Find the database sequence with the most matching k-mers.
        # This is slow because it is doing random memory writes in a counts
        # array that is too large to fit in CPU cache.
        from numpy import zeros, int32
        counts = zeros((self.num_sequences,), int32)
        for seqi in seqi_list:
            counts[seqi] += 1
        database_sequence_number = counts.argmax()
        num_kmer_matches = counts[database_sequence_number]

        return database_sequence_number, num_kmer_matches

    def kmer_sequences(self, kmer_list, nthreads = 1):
        '''
        For each k-mer return an array of indices of sequences that contain that k-mer.
        '''
        if self.indices_in_memory:
            seqi_list = []
            for kmer in kmer_list:
                o,c = self.kmer_sequence_offsets[kmer], self.counts[kmer]
                seqi_list.append(self.kmer_seq_indices[o:o+c])
            return seqi_list

        uint_size = 4
        if nthreads <= 1:
            f = self.sequence_indices_file
            bdata = []
            for kmer in kmer_list:
                o,c = self.kmer_sequence_offsets[kmer], self.counts[kmer]
                f.seek(uint_size * offset)
                bdata.append(f.read(uint_size * count))
        else:
            # Do a multi-threaded read from the kmer sequence file.
            seqi_path = self.file_paths()[3]
            blocks = tuple(zip(uint_size*self.kmer_sequence_offsets[kmer_list],
                               uint_size*self.counts[kmer_list]))
            bdata = read_blocks_threads(seqi_path, blocks, nthreads=nthreads)

        # Convert the read bytes to numpy arrays of sequences indices.
        from numpy import frombuffer, uint32
        seqi_list = [frombuffer(data, dtype = uint32) for data in bdata]

        return seqi_list

    def title_and_sequence(self, seq_num):
        offsets = self.fasta_sequence_offsets
        with open(self._sequences_path, 'rb') as f:
            f.seek(offsets[seq_num])
            entry = f.read(self.sizes[seq_num]).decode('utf-8')
        title, sequence = entry.split('\n')[:2]
        return title, sequence

    def file_paths(self):
        return index_file_paths(self._sequences_path)

    def load_index(self, index_in_memory = False):
        self.indices_in_memory = index_in_memory
        size_path, sizes_path, counts_path, seqs_path = self.file_paths()
        from numpy import fromfile, uint32, uint16
        self.counts = fromfile(counts_path, dtype=uint32)
        self.sizes = fromfile(sizes_path, dtype=uint16)
        if index_in_memory:
            self.kmer_seq_indices = fromfile(seqs_path, dtype=uint32)
        with open(size_path, 'r') as fs:
            import json
            s = json.load(fs)
            self.k = s['k']
            self.num_sequences = s['num_sequences']

    @property
    def sequence_indices_file(self):
        f = getattr(self, '_seq_indices_file', None)
        if f is None:
            seqs_path = self.file_paths()[3]
            self._seq_indices_file = f = open(seqs_path, 'rb')
        return f
    
    @property
    def kmer_sequence_offsets(self):
        offsets = getattr(self, '_kmer_seq_offsets', None)
        if offsets is None:
            from numpy import cumsum, uint64
            offsets = cumsum(self.counts, dtype=uint64)
            offsets -= self.counts
            self._kmer_seq_offsets = offsets
        return offsets
    
    @property
    def fasta_sequence_offsets(self):
        offsets = getattr(self, '_fasta_seq_offsets', None)
        if offsets is None:
            from numpy import cumsum, uint64
            offsets = cumsum(self.sizes, dtype=uint64)
            offsets -= self.sizes
            self._fasta_seq_offsets = offsets
        return offsets

amino_acid_characters = 'ACDEFGHIKLMNPQRSTVWY'

def number_of_kmers(k):
    return len(amino_acid_characters)**k

aaindex = None		# Map amino acid ascii integer to integer in range 0-19
def sequence_as_integer_array(sequence):
    global aaindex
    if aaindex is None:
        from numpy import zeros, uint8
        aaindex = zeros((256,), uint8)
        for i,c in enumerate(amino_acid_characters):
            aaindex[ord(c)] = i

    # Sequence with characters replaced by integers 0-19.
    from numpy import frombuffer, uint8
    seqi = aaindex[frombuffer(sequence.encode('ascii'), uint8)]
    return seqi
            
def sequence_kmers(sequence, k, unique = True, kmer_range = None):
    '''
    Return the kmers in the given sequence.
    They are retuned as an array of integers.
    Each k-mer is represented as its integer index.
    '''
    # Convert sequence characters to array of integers
    seqi = sequence_as_integer_array(sequence)

    # Compute indices with array operations for speed.
    from numpy import zeros, uint32
    kmers = zeros((len(sequence)-k+1,), uint32)
    m = len(kmers)
    naa = len(amino_acid_characters)
    for i in range(k):
        kmers[:] *= naa
        kmers[:] += seqi[i:m+i]

    # Include only kmers in a range
    if kmer_range is not None:
        kmin, kmax = kmer_range
        in_range = (kmers >= kmin) & (kmers < kmax)
        kmers = kmers[in_range] - kmin

    # Remove duplicate kmers
    if unique:
        from numpy import unique
        kmers = unique(kmers)  # This also sorts the indices as a side-effect

    return kmers

def index_file_paths(sequences_path):
    from os.path import splitext
    basename = splitext(sequences_path)[0]
    return basename + '.size', basename + '.sizes', basename + '.counts', basename + '.seqs'

def read_blocks(path, blocks):
    data = []
    with open(path, 'rb') as f:
        for offset, bytes in blocks:
            f.seek(offset)
            b = f.read(bytes)
            data.append(b)
    return data

def read_blocks_list(path, blocks, read_data, offset):
    data = read_blocks(path, blocks)
    for i,bdata in enumerate(data):
        read_data[offset+i] = bdata

def read_blocks_threads(path, blocks, nthreads = 1):
    from math import ceil
    batch_size = int(ceil(len(blocks) / nthreads))
    data = [None] * len(blocks)
    threads = []
    import threading
    for t in range(nthreads):
        bstart = t*batch_size
        tblocks = blocks[bstart:bstart+batch_size]
        rt = threading.Thread(target=read_blocks_list, args=(path, tblocks, data, bstart))
        rt.start()
        threads.append(rt)
    for rt in threads:
        rt.join()
    return data


def accession_codes(fasta_title_line, database_version):
    # Example title line: >MGYP000887593774 FL=1 CR=1
    title = fasta_title_line
    return {'mgnify id': title[1:17],
            'db version': database_version}

def clean_sequence(sequence):
    '''Remove characters that are not one of 20 standard amino acids.'''
    aset = set(amino_acid_characters)
    return ''.join(c for c in sequence if c in aset)

def search_database():
    try:
        sequences = parse_request_sequences()
        matches = sequence_search(sequences)
        report_matches({'sequences':matches})
    except Exception:
        import sys, json, traceback
        error_msg = json.dumps({'error': traceback.format_exc()})
        sys.stdout.write('Content-Type: application/json\n\n%s' % error_msg)

if __name__ == '__main__':
    search_database()
