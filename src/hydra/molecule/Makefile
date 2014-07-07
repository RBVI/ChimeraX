TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/molecule

PYSRCS = __init__.py align.py biomt.py blastpdb.py colorcmd.py colors.py connect.py \
	fetch_pdb.py fetch_uniprot.py mmcif.py mol_session.py molecule.py pdb.py \
	residue_codes.py showcmd.py

DATA = bond_templates

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
