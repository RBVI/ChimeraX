TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS =

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera

PYSRCS = $(wildcard *.py)

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

$(SUBDIR_INSTALL): dir

lint:
	pyflakes $(PYSRCS)

clean:
	rm -rf __pycache__
