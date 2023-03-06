# Check percent sequence identity needed to get a certain number of k-mer matches.
# See how it depends on sequence length.

def kmer_count(seq_length, mutations, k = 5):
    '''
    See what percent identity is needed for a given sequence length
    to get a certain minimum number of matching kmers.
    '''
    from random import sample
    pos = range(seq_length)
    mpos = sample(pos, mutations)
    mpos.sort()
    apos = [-1] + mpos + [seq_length]
    n = 0
    for i in range(len(apos)-1):
        n += max(0, apos[i+1]-apos[i]-k)
    return n

from statistics import median
slengths = [50, 100, 200, 500, 1000, 2000]
nsample = 1000
min_matches = 15
for slen in slengths:
    for m in range(0, slen):
        kc = [kmer_count(slen, m) for i in range(nsample)]
        kc.sort()
#        kmers = median(kc)
        kmers = kc[10]
        if kmers <= min_matches:
            print ('seq length %d, %d matches, percent identity %.1f%%' % (slen, kmers, 100*(slen-m)/slen))
            break

'''
# Percent identity thqt gives 50% chance of getting 15 matches.

$ python3 kmer_counts.py 
seq length 50, 15 matches, percent identity 80.0%
seq length 100, 13 matches, percent identity 69.0%
seq length 200, 15 matches, percent identity 60.5%
seq length 500, 15 matches, percent identity 50.8%
seq length 1000, 15 matches, percent identity 44.0%
seq length 2000, 15 matches, percent identity 38.4%

# 90% chance
seq length 50, 15 matches, percent identity 84.0%
seq length 100, 15 matches, percent identity 74.0%
seq length 200, 15 matches, percent identity 66.0%
seq length 500, 15 matches, percent identity 54.8%
seq length 1000, 15 matches, percent identity 47.1%
seq length 2000, 15 matches, percent identity 41.8%

# 99% channce (100 samples)
seq length 50, 14 matches, percent identity 86.0%
seq length 100, 14 matches, percent identity 78.0%
seq length 200, 14 matches, percent identity 70.5%
seq length 500, 14 matches, percent identity 60.0%
seq length 1000, 15 matches, percent identity 52.1%
seq length 2000, 14 matches, percent identity 46.0%

# 99% chance (1000 samples)
seq length 50, 14 matches, percent identity 86.0%
seq length 100, 15 matches, percent identity 78.0%
seq length 200, 15 matches, percent identity 68.0%
seq length 500, 14 matches, percent identity 57.6%
seq length 1000, 15 matches, percent identity 50.4%
seq length 2000, 15 matches, percent identity 44.0%
'''
