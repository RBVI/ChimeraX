TOP= ../../..
include $(TOP)/mk/config.make

PYSRCS= $(wildcard *.py)
PKG_DIR= $(PYSITEDIR)/chimerax/core/ui
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/ui
DATA = splash.jpg

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)
	#if [ -d "$(APP_PKG_DIR)" ]; then $(RSYNC) $(PYSRCS) $(APP_PKG_DIR); fi

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	#if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

clean:
	rm -rf __pycache__

lint:
	$(PYLINT) $(PYSRCS)
