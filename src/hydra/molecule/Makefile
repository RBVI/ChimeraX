TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/molecule

PYSRCS = __init__.py align.py colors.py connect.py gridsurf.py mol_session.py molecule.py showcmd.py

DATA = bond_templates

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)

clean:
	rm -rf __pycache__
