TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/file_io

PYSRCS = biomt.py blastpdb.py collada.py fetch.py fetch_eds.py fetch_emdb.py fetch_pdb.py \
	fileicon.py history.py mmcif.py opensave.py \
	pdb.py read_apr.py read_stl.py read_swc.py session_file.py \
	write_stl.py

SUBDIRS	= SessionUtil
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(SUBDIR_INSTALL): $(PKG_DIR)

clean:
	rm -rf __pycache__
