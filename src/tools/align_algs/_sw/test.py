# The "_sw" module contains a C++ implementation of the
# Smith-Waterman alignment algorithm.  It is basically
# a straight translation of the Python code from the
# Dynamic Programming homework.
#
# There are two functions defined in "_sw": "score"
# and "align".  Both functions take the same set of
# arguments, but "score" only computes the score of
# the best alignment, while "align" returns both
# the best score and the corresponding alignment.
#
# The functions take five arguments in the following order:
#	seq1		first sequence (string)
#	seq2		second sequence (string)
#	matrix		a similarity matrix (see below)
#	gapOpen		gap opening penalty (floating point)
#	gapExtension	gap extension penalty (floating point)
#
# The similarity matrix is actually stored as a dictionary
# whose keys are two-tuples of residue names, and whose values
# are the similarity values.  The residue names must be
# single-character strings, and the similarity values must be 
# floating point numbers.
#
# "score" returns the best score as a floating point number.
# "align" returns the best score and the corresponding
# alignment as a two-tuple.  The score is a floating point
# number.  The alignment is a two-tuple of strings, where
# the first and second strings represent bases (and gaps)
# from sequences "seq1" and "seq2", respectively.
#
# The code below demonstrates how the module might be used.
#
from chimerax.seqalign.align_algs import SmithWaterman

matrix = {
	('A', 'A'): 1.0,
	('A', 'C'): -1.0/3.0,
	('A', 'G'): -1.0/3.0,
	('A', 'U'): -1.0/3.0,
	('C', 'A'): -1.0/3.0,
	('C', 'C'): 1.0,
	('C', 'G'): -1.0/3.0,
	('C', 'U'): -1.0/3.0,
	('G', 'A'): -1.0/3.0,
	('G', 'C'): -1.0/3.0,
	('G', 'G'): 1.0,
	('G', 'U'): -1.0/3.0,
	('U', 'A'): -1.0/3.0,
	('U', 'C'): -1.0/3.0,
	('U', 'G'): -1.0/3.0,
	('U', 'U'): 1.0,
}

seq1 = 'AAUGCCAUUGACGG'
seq2 = 'CAGCCUCGCUUAG'
score = SmithWaterman.score(seq1, seq2, matrix, 1, 0.3333)
print('Aligning:')
print(seq1)
print(seq2)
print('Best score is', score)
print()

#seq1 = 'UGCCGCUGACGG'
#seq2 = 'GCCCUGCUUAG'
seq1 = 'GCCGCGACGG'
seq2 = 'GCCCGCAG'
score, alignment = SmithWaterman.align(seq1, seq2, matrix, 1, 0.3333)
print('Aligning:')
print(seq1)
print(seq2)
print('Best score is', score)
print('Corresponding alignment is:')
print(alignment[0])
print(alignment[1])
