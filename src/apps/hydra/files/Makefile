TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/files

PYSRCS = fetch.py fileicon.py history.py opensave.py \
	read_apr.py read_swc.py session_file.py

SUBDIRS	= SessionUtil
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(SUBDIR_INSTALL): $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
