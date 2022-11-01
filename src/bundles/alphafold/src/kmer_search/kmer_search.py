def sequence_search(sequences, database_path, min_kmer_matches = 15, nthreads = 1):
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
        seq_num, num_kmer_matches = kmer_index.search(sequence, nthreads = nthreads)
        if num_kmer_matches < min_kmer_matches:
            results.append(None)
        title, db_sequence = kmer_index.title_and_sequence(seq_num)
        uid, uname = uniprot_id_and_name(title)
        results.append((uid, uname, db_sequence, num_kmer_matches))
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

ski_func = None
ski_indices = None
ski_ind_p = None
def c_sequence_kmers(sequence, k, unique = True, kmer_range = None):
    '''
    This is a version of the Python sequence_kmers() routine coded in C for speed.
    It turns out it didn't speed up much, about a factor of two, apparently because
    the ctypes overhead to call the function takes half the of the Python version.
    '''
    global ski_func, ski_indices, ski_ind_p
    if ski_func is None:
        from sys import platform
        lib_path = './make_kmer_index.dylib' if sys.platform == 'darwin' else './make_kmer_index.so'
        import ctypes
        lib = ctypes.PyDLL(lib_path)
        ski_func = lib.sequence_kmers
        ski_func.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        ski_func.restype = ctypes.c_int

    if ski_indices is None:
        max_size = 2**16
        from numpy import empty, uint32
        ski_indices = empty((max_size,), uint32)
        ski_ind_p = ski_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))

    seq_p = sequence.encode('ascii')
    kmin, kmax = (0, number_of_kmers(k)) if kmer_range is None else kmer_range
    uniq = 1 if unique else 0
    count = ski_func(seq_p, len(sequence), k, uniq, kmin, kmax, ski_ind_p)
    return ski_indices[:count]

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

def uniprot_id_and_name(fasta_title_line):
    title = fasta_title_line
    if title.startswith('>AFDB:'):
        uniprot_id, uniprot_name = afdb_v3_uniprot_id_and_name(title)
    elif '|' in title:
        # AlphaFold Database version 2 title line format
        # >tr|X1WFM8|X1WFM8_DANRE EPS8-like 2 OS=Danio rerio OX=7955 GN=eps8l2 PE=3 SV=1
        fields = title.split('|')
        uniprot_id = fields[1] if len(fields) >= 2 else 'unknown'
        uniprot_name = fields[2].split()[0] if len(fields) >= 3 else 'unknown'
    else:
        uniprot_id = uniprot_name = 'unknown'
    return uniprot_id, uniprot_name

def afdb_v3_uniprot_id_and_name(title):
    # AlphaFold Database version 3 title line format
    # >AFDB:AF-A0A2L2JPH6-F1 Uncharacterized protein UA=A0A2L2JPH6 UI=A0A2L2JPH6_9NOCA OS=Nocardia cyriacigeorgica OX=135487 GN=C5B73_08745
    uniprot_id = uniprot_name = 'unknown'
    fields = title.split()
    for f in fields:
        if f.startswith('UA='):
            uniprot_id = f[3:]
        if f.startswith('UI='):
            uniprot_name = f[3:]
    return uniprot_id, uniprot_name

def create_index_files(seqs_path, k = 5, unique = True, max_memory = 0):
    counts, num_sequences = kmer_counts(seqs_path, k)
    sizes = sequence_entry_sizes(seqs_path, num_sequences)
    size_path, sizes_path, counts_path, seqi_path = index_file_paths(seqs_path)
    with open(size_path, 'w') as fs:
        import json
        json.dump({'k': k, 'num_sequences': num_sequences}, fs)
    sizes.tofile(sizes_path)
    counts.tofile(counts_path)

    if max_memory is None:
        kmer_seq_indices = kmer_sequences(seqs_path, k, counts)
        kmer_seq_indices.tofile(seqi_path)
    else:
        from numpy import uint64
        from math import ceil
        index_memory = 4 * counts.sum(dtype = uint64)
        nblocks = int(ceil(index_memory / max_memory))
        kblock = int(ceil(number_of_kmers(k) / nblocks))
        print (f'Saving kmer map in {nblocks} blocks')
        with open(seqi_path, 'wb') as sf:
            for kmin in range(0, number_of_kmers(k), kblock):
                kmer_seq_indices = kmer_sequences(seqs_path, k, counts,
                                                  kmer_range = (kmin,kmin+kblock))
                kmer_seq_indices.tofile(sf)
                kmer_seq_indices = None	# Release memory

def c_create_index_files(sequences_path, k = 5, unique = True, max_memory = 0):
    from sys import platform
    lib_path = './make_kmer_index.dylib' if platform == 'darwin' else './make_kmer_index.so'
    import ctypes
    lib = ctypes.PyDLL(lib_path)
    cif_func = lib.create_index_files
    cif_func.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_long)
    cif_func.restype = None
    seq_p = sequences_path.encode('ascii')
    uniq = 1 if unique else 0
    cif_func(seq_p, k, uniq, max_memory)

def sequence_entry_sizes(seqs_path, nseq):
    from numpy import empty, uint16
    sizes = empty((nseq,), uint16)

    with open(seqs_path, 'r') as f:
        for i in range(nseq):
            title = f.readline()
            seq = f.readline()
            sizes[i] = len(title) + len(seq)

    return sizes
    
def kmer_counts(seqs_path, k):
    from numpy import zeros, uint32, uint64, empty
    counts = zeros((number_of_kmers(k),), uint32)

    with open(seqs_path, 'r') as f:
        nseq = 0
        while True:
            line = f.readline().strip()
            if line == '':
                break
            if line.startswith('>'):
                continue
            ki = sequence_kmers(line, k)
#            ki = c_sequence_kmers(line, k)
            counts[ki] += 1
            nseq += 1

    return counts, nseq

def kmer_sequences(seqs_path, k, counts, kmer_range = None):
    '''
    Return numpy array of sequence indices of sequences containing each kmer.
    '''
    if kmer_range is not None:
        counts = counts[kmer_range[0]:kmer_range[1]]
        
    from numpy import uint64, empty, uint32
    offsets = counts.cumsum(dtype = uint64)
    offsets -= counts
    ni = counts.sum(dtype = uint64)
    kmer_seq_indices = empty((ni,), uint32)
    counts[:] = 0

    with open(seqs_path, 'r') as f:
        nseq = 0
        while True:
            line = f.readline().strip()
            if line == '':
                break
            if line.startswith('>'):
                continue
            ki = sequence_kmers(line, k, kmer_range=kmer_range)
#            ki = c_sequence_kmers(line, k, kmer_range=kmer_range)
            o = offsets[ki] + counts[ki]
            kmer_seq_indices[o] = nseq
            counts[ki] += 1
            nseq += 1

    return kmer_seq_indices

def mutate(sequence, frac):
    from random import random, sample
    return ''.join(sample(amino_acid_characters, 1)[0] if random() <= frac else c
                   for c in sequence)

def percent_identity(seq1, seq2):
    n = min(len(seq1), len(seq2))
    same = len([i for i in range(n) if seq1[i] == seq2[i]])
    return 100 * same / n

def read_speed(seqs_path, k=5):
    '''Time reading sequences and calculating kmers.'''
    from time import time
    t0 = time()
    kmer_counts(seqs_path, k)
    t1 = time()
    print ('read %d sequences and counted kmers in %.3f seconds' % (nseq, t1-t0))

def report_kmer_index_info(kmer_index):
    k = kmer_index.k
    nk = number_of_kmers(k)
    kmer_index.load_index()
    num_seq = kmer_index.num_sequences
    nf = (kmer_index.counts == 0).sum()
    from numpy import uint64
    snk = kmer_index.counts.sum(dtype = uint64)
    print (f'read {k}-mer map with {nk} entries ({nf} with no sequences) for {num_seq} sequences containing {snk} {k}-mers')
    kmer_counts = kmer_index.counts.copy()
    kmer_counts.sort()
    kmer_counts = kmer_counts[::-1]
    print(f'most common {k}-mers: {kmer_counts[:10]}')

def search_mutated_sequences(sequence, kmer_index, nthreads = 1):
    from numpy import arange
    for frac in arange(0, .8, .05):
        seq = mutate(sequence, frac)
        pid = percent_identity(seq, sequence)
        s, matches = kmer_index.search(seq, nthreads = nthreads)
        title, db_sequence = kmer_index.title_and_sequence(s)
        uid, uname = uniprot_id_and_name(title)
        print(f'%{int(pid)} identity found sequence {uname} ({s}) with {matches} matching {kmer_index.k}-mers')

def clean_sequence(sequence):
    '''Remove characters that are not one of 20 standard amino acids.'''
    aset = set(amino_acid_characters)
    return ''.join(c for c in sequence if c in aset)

def test_search(seqs_path):
    #sequence = 'MTDDKAGPSGLSLKEAEEIHSYLIDGTRVFGAMALVAHILSAIATPWLG'
    sequence = 'MIRSKEPHNNLCLLYNQGLMPYLDAHRWQRSLLNERIHDPSLDDVLILLEHPPVYTLGQGSNSDFIKFDIDQGEYDVHRVERGGEVTYHCPGQLVGYPILNLQRYRKDLHWYLRQLEEVIIRVLTVYGLQGERIPAFTGVWLQGRKVAAIGIKVSRWITMHGFALNVCPDMKGFERIVPCGISDKPVGSLAEWIPGITCQEVRFYVAQCFAEVFGVELIESQPQDFFRPE'
    #sequence = 'MAARTTAVGTPSWITAEKENALQLLQNEKEEVVYPAQHQLEWLNEHMAEVFSRSHLRADSNSDVANVFKTPGKMRGKTPRTGRKRNAQDPRIPLSDVFSSKPLQRHSPQKQPSPTKLRFHVSADQERQTYKSVTDSGYHTASQDDVDVDISSHVQPVQPTPTAKEAPSENSDVISEHDMEDVRDEGHRTTEGSFHSAKEDQTTKIAGPGAVSLEEVTRTDPQSNRQSPQSAPLETMHSQEPHDLDELGSPSDASTPDRPLVRKGSLTFASLPPRDPLKSSIGARISRTSHIDQGRQHGSIVHGARQTLPPHQALPKPPQNLDADIPMHDVNDEDDDSNVQILGHDSDAESQTIRDHSEASTKRLHEKIDMLGKAAAPRPSKSIPAAIPLPDSTQAVEHDDQVARPALQVQAMQDDEDDWIRPLTSPNAAQSSPAKPANLEDSDEDEFDCRAPELIAHEERMRTPVRMSPGPGKMMPGFGHTKSASTATLASPAKAAMAPPASPAKSVSVSNPAQATTTPQGSPRRYLDLSASKSKLQSIMKTAKGLFTSSASVSAAAKLETLSPNSLKTASSAMPGMYPSLYSVIEDKALPSNPPKESRKTRSSSERDKEERRKEKETQLTQAMDAQLEKVRAQERKKAEDQKLARERTAKKDAQQPPSTSPKPQAEEPQHGPSELPARPTRPVRLGREHPVNKAKPAPVSIRVGTLSQRMPVNTGPNAQDSLAAEPKRPGLNKKASSASLQSTTSTVLKTSVAGPPPKPRALLAAERKKEQDEREAQRKLEQKRELERKRVVQQEEARKQEQKQRVEAEKRERERVAAEQAKRQAQQQAIERKRQEAARKAEQQRLDRAANEAAQTRPPSRVAAQNAGRPLLNHPLPTNPAKPAKRPLEEEANGSRPQNSKYGNGMLQGDAKRRKTEEECLIESAPRPVVSGAPIRQSQLGKKASFLTHPSYVQTQPSTHAGQFPQPPSRVAPPQMQQYATGGKIPFADAPHPPNHAKTPVSIMQHKTIQTVKSSPQYPNGESIHLPEIPTDSEDEDSDDGGSGFAIPDWATPGHLTEQLIRQEGMDGDAVFGPIAPLKIEEIFSKGNKDRLKRLRDRTSSANWALSGDGLTLEEVRADREQRERMRLQGGWRYGK'

    kmer_index = KmerSequenceIndex(seqs_path)
    #report_kmer_index_info(kmer_index)
    search_mutated_sequences(sequence, kmer_index, nthreads = 16)
    '''
    for i in range(100):
        s, matches = kmer_index.search(mutate(sequence,1), nthreads = 1)
        print (s, matches)
    '''

if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3 or argv[1] not in ['makeindex', 'test']:
        print('Syntax: python3 kmer_search.py test|makeindex sequences.fasta')
    else:
        sequences_path = argv[2]
        if argv[1] == 'test':
            test_search(sequences_path)        
        elif argv[1] == 'makeindex':
            try:
                c_create_index_files(sequences_path, max_memory = 16 * 1024 * 1024 * 1024)
                # Took 3 hours for 214M seqs
            except:
                create_index_files(sequencess_path, max_memory = 16 * 1024 * 1024 * 1024)
