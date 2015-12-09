TOP= ../../..
include $(TOP)/mk/config.make

PYSRCS= $(wildcard *.py)
PKG_DIR= $(PYSITEDIR)/chimerax/core/webservices
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/webservices

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	#if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__
