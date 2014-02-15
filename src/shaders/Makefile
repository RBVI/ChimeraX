TOP = ../..
include $(TOP)/mk/config.make

PKG_DIR = $(datadir)/shaders

install:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	cp -p *.txt $(PKG_DIR)

clean:
