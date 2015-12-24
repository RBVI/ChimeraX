TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS = cellpack cmd_line contacts exp_cmd help_viewer \
	  log map_series_gui model_panel mouse_modes movie \
	  shell shortcuts sideview title test toolshed

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax

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
