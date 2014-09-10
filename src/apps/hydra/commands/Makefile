TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/commands

PYSRCS = cameracmd.py commands.py cyclecmd.py lightcmd.py materialcmd.py motion.py parse.py \
	shortcuts.py silhouettecmd.py turncmd.py

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(SUBDIR_INSTALL): $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
