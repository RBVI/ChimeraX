TOP= ../../..
include $(TOP)/mk/config.make

PYSRCS= $(wildcard *.py)
PKG_DIR= $(PYSITEDIR)/chimera/core/toolshed
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/toolshed
DATA = helpdir

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)
	#if [ -d "$(APP_PKG_DIR)" ]; then $(RSYNC) $(PYSRCS) $(APP_PKG_DIR); fi

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	#if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__
