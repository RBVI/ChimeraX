# Test speed of reading using multiple threads to get 1000 blocks each of 10K size
# at random locations in a 100GB file.

# This code differs from read_threads.py in that it does not open a separate copy
# of the file in each thread.  Instead it uses a single open copy of the file and
# the pread() function.  Using pread() avoids doing f.seek() in different threads.

# Unfortunately this pread() multi-threaded read gets no speed-up on Wynton with
# the beegfs file system.  It may be that Python does not allow multiple pread()
# calls to execute simultaneously because it holds the global interpretter log.
# Or it could be that operating system does not allow simultaneous pread() calls.

from os import pread
def read_blocks(fd, blocks):
    data = [pread(fd, bytes, offset) for offset, bytes in blocks]
    return data

def read_blocks_list(fd, blocks, list, element):
    list[element] = read_blocks(fd, blocks)

def read_blocks_threads(fd, blocks, nthreads = 1):
    import threading
    from math import ceil
    batch_size = int(ceil(len(blocks) / nthreads))
    threads = []
    data = [None] * nthreads
    for t in range(nthreads):
        bstart = t*batch_size
        rt = threading.Thread(target=read_blocks_list, args=(fd, blocks[bstart:bstart+batch_size], data, t))
        rt.start()
        threads.append(rt)
    for rt in threads:
        rt.join()
    d = sum(data, [])
    return d

#path = '/wynton/home/ferrin/goddard/nobackup/afdb/alphafold214M.fasta'
path = '/wynton/home/ferrin/goddard/nobackup/afdb100/alphafold100M.fasta'
file = open(path, 'rb')
fd = file.fileno()
size = 99265850384	# bytes
block_size = 16*1024    # bytes
num_blocks = 1024

max_offset = size - block_size
from random import randint
blocks = [(randint(0,max_offset), block_size) for b in range(num_blocks)]
from time import time
t0 = time()
#data = read_blocks(fd, blocks)
data = read_blocks_threads(fd, blocks, 8)
t1 = time()
elapsed = '%.2f' % (t1-t0)
rbytes = sum([len(d) for d in data])
print(f'read {num_blocks} blocks of {block_size} bytes each in {elapsed} seconds, got total of {rbytes} bytes')
