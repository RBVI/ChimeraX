#!/bin/env python3
# Test speed of reading using multiple threads to get 1000 blocks each of 10K size
# at random locations in a 100GB file.

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

def read_test(path, num_threads = 1, num_blocks = 1024, block_size = 16*1024):
    from os.path import getsize
    size = getsize(path)	# bytes
    max_offset = size - block_size
    from random import randint
    blocks = [(randint(0,max_offset), block_size) for b in range(num_blocks)]

    from time import time
    t0 = time()
    if num_threads == 0:
        data = read_blocks(path, blocks)
    else:
        data = read_blocks_threads(path, blocks, num_threads)
    t1 = time()
    return data, t1-t0

if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) <= 1:
        print('Syntax: read_threads.py <path> [num-threads] [num-blocks] [block-size]')
        sys.exit(0)
    path = args[1]
    num_threads = int(args[2]) if len(args) >= 3 else 1
    num_blocks = int(args[3]) if len(args) >= 4 else 1024
    block_size = int(args[4]) if len(args) >= 5 else 16*1024
    data, time = read_test(path, num_threads, num_blocks, block_size)
    read_bytes = sum([len(d) for d in data])
    tsec = '%.03f' % time
    print(f'read {num_blocks} blocks of {block_size} bytes each, got total of {read_bytes} bytes in {tsec} seconds')
