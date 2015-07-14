TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS = cmd_line contacts log map_series_gui mouse_modes movie shell sideview toolshed

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera

PYSRCS = $(wildcard *.py)

all: $(PYOBJS)

install: all dir
ifneq (,$(PYSRCS))
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
endif

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

$(SUBDIR_INSTALL): dir

lint:
ifneq (,$(PYSRCS))
	$(PYLINT) $(PYSRCS)
endif

clean:
	rm -rf __pycache__
