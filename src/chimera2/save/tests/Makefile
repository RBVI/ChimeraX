HEADLESS_CHIMERA = /usr/local/chimera-1.6.1-headless/bin/chimera

all:

clean:
	rm -f *.pyc

atoms: pdb3cc4_atoms.py pdb3fx2_atoms.py pdb3k9f_atoms.py pdbmtx_atoms.py

pdb3cc4_atoms.py: extract.py
	$(HEADLESS_CHIMERA) pdb:3cc4 extract.py

pdb3fx2_atoms.py: extract.py
	$(HEADLESS_CHIMERA) pdb:3fx2 extract.py

pdb3k9f_atoms.py: extract.py
	$(HEADLESS_CHIMERA) pdb:3k9f extract.py

pdbmtx_atoms.py: extract.py
	$(HEADLESS_CHIMERA) ~gregc/mtx.pdb extract.py
