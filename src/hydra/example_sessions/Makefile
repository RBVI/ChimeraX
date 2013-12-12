TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/example_sessions
DATA_FILES = 6bna.hy 6bna.jpg Tropomyosin.hy Tropomyosin.jpg \
	     XrayDensity.hy XrayDensity.jpg Reovirus.hy Reovirus.jpg \
	     sessions

install: all pkg_dir
	$(RSYNC) $(DATA_FILES) $(PKG_DIR)

clean:
