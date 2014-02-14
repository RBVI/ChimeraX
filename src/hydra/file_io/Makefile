TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/file_io

PYSRCS = biomt.py fetch.py fetch_eds.py fetch_emdb.py fetch_pdb.py \
	fileicon.py history.py opensave.py \
	pdb.py read_apr.py read_stl.py read_swc.py session_file.py \
	write_stl.py

SUBDIRS	= SessionUtil
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all pkg_dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(SUBDIR_INSTALL): pkg_dir

clean:
	rm -rf __pycache__
