TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/example_sessions
DATA_FILES = 6bna.hy 6bna.jpg Tropomyosin.hy Tropomyosin.jpg \
	     XrayDensity.hy XrayDensity.jpg Reovirus.hy Reovirus.jpg \
	     sessions

install: all $(PKG_DIR)
	$(RSYNC) $(DATA_FILES) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
