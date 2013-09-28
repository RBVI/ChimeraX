TOP = ../../..
include $(TOP)/mk/config.make

SUBDIRS	= SessionUtil

PKG_DIR = $(PYSITEDIR)/hydra/file_io

PYSRCS = biomt.py fetch.py fetch_emdb.py fetch_pdb.py history.py opensave.py \
	pdb.py read_apr.py read_stl.py read_swc.py session.py

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
