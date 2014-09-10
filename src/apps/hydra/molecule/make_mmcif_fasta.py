from hydra.molecule import blastpdb
fa = blastpdb.create_fasta_database('/usr/local/mmCIF')
f = open('/Users/goddard/Desktop/mmcif.fasta', 'w')
f.write(fa)
f.close()
# To make NCBI blast+ database from fasta file use command:
# 	makeblastdb -in mmcif.fasta -out mmcif -dbtype prot
