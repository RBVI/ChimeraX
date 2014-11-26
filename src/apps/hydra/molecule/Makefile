TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/molecule

SUBDIRS	= molecule_cpp

PYSRCS = __init__.py align.py ambient.py biomt.py blastpdb.py colors.py connect.py \
	fetch_pdb.py fetch_uniprot.py mass.py mcommand.py mmcif.py mol_session.py molecule.py pdb.py \
	residue_codes.py

DATA = bond_templates

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
