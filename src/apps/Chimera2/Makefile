# Use this make file to make a frozen verson of chimera for debugging

TOP	= ../../..
include $(TOP)/mk/config.make

# Freeze source was copied from tmp/Python-*/Tools/freeze
FREEZE_SOURCE = $(datadir)/freeze

ifeq ($(OS),Linux)
LDFLAGS	+= -Wl,-rpath,'$$ORIGIN/../lib'
endif
ifeq ($(OS),Darwin)
LDFLAGS	+= -Wl,-rpath,'@executable_path/../lib'
endif

PROG	= $(APP_NAME)
STRIP	=
BINMODE	= 755

PY_MAIN = $(PROG)_main.py

ICONS	= Chimera2-icon.svg Chimera2-icon512.png
ifeq ($(OS),Linux)
LINUX_ICONS = Chimera2-icon128.png Chimera2-icon64.png Chimera2-icon32.png Chimera2-icon16.png
ICONS	+= $(LINUX_ICONS)
endif
GEN_SRCS= frozen.c M___main__.c
OBJS	= $(GEN_SRCS:.c=.o)
INCS	+= $(PYTHON_INCLUDE_DIRS)
LIBS	+= $(PYTHON_LIB)

all: $(PROG)

install: $(PROG)
	$(RSYNC) $(PROG) $(bindir)
	$(RSYNC) $(PY_MAIN) $(PYSITEDIR)
	$(RSYNC) $(ICONS) $(datadir)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) $(PROG) $(app_bindir); \
		$(RSYNC) $(PY_MAIN) $(APP_PYSITEDIR); \
		$(RSYNC) $(ICONS) $(app_datadir); \
	fi
ifeq ($(OS),Darwin)
	-if [ -d "$(app_prefix)" ]; then \
		ln -nfs ../bin/$(PROG) $(app_prefix)/MacOS/$(APP_NAME); \
		$(RSYNC) Info.plist $(app_prefix)/Info.plist; \
		$(RSYNC) chimera-icon.icns $(app_prefix)/Resources/chimera-icon.icns; \
	fi
endif

clean:
	rm -f $(PROG) $(GEN_SRCS) $(OBJS) frozen.c.orig __main__.py gensrcs \
		Info.plist $(LINUX_ICONS)

lint:
	$(PYLINT) *.py

run:
	$(PYTHON_EXE) chimera2_main.py

run-nogui:
	$(PYTHON_EXE) chimera2_main.py --nogui

$(PROG): $(OBJS)
	$(PROG_LINK)

__main__.py: $(PY_MAIN)
	$(PYTHON_EXE) mk__main__.py $<

frozen.c M___main__.c: gensrcs

gensrcs: __main__.py
	$(PYTHON_EXE) mkfrozen.py $(FREEZE_SOURCE) $<
	patch -b -V simple < frozen.patch
	touch gensrcs

Chimera2-icon512.png: Chimera2-icon.svg
	inkscape --export-png=Chimera2-icon512.png --export-width=512 --export-background-opacity=0 --without-gui Chimera2-icon.svg


ifeq ($(OS),Darwin)
install: Info.plist
endif

Info.plist: mkinfo.py chimera-icon.icns
#	$(bindir)/$(PROG) --nogui mkinfo.py $(APP_NAME) || rm Info.plist
	$(PYTHON_EXE) mkinfo.py $(APP_NAME) || rm Info.plist

chimera-icon.icns:	Chimera2-icon512.png
	mkdir chimera-icon.iconset
	sips -z 16 16     Chimera2-icon512.png --out chimera-icon.iconset/icon_16x16.png
	sips -z 32 32     Chimera2-icon512.png --out chimera-icon.iconset/icon_16x16@2x.png
	sips -z 32 32     Chimera2-icon512.png --out chimera-icon.iconset/icon_32x32.png
	sips -z 64 64     Chimera2-icon512.png --out chimera-icon.iconset/icon_32x32@2x.png
	sips -z 128 128   Chimera2-icon512.png --out chimera-icon.iconset/icon_128x128.png
	sips -z 256 256   Chimera2-icon512.png --out chimera-icon.iconset/icon_128x128@2x.png
	sips -z 256 256   Chimera2-icon512.png --out chimera-icon.iconset/icon_256x256.png
	sips -z 512 512   Chimera2-icon512.png --out chimera-icon.iconset/icon_256x256@2x.png
	cp Chimera2-icon512.png chimera-icon.iconset/icon_512x512.png
	iconutil -c icns chimera-icon.iconset
	rm -rf chimera-icon.iconset

ifeq ($(OS),Linux)
install: $(LINUX_ICONS)
endif

Chimera2-icon128.png:
	convert Chimera2-icon512.png -geometry 128x128 Chimera2-icon128.png

Chimera2-icon64.png:
	convert Chimera2-icon512.png -geometry 64x64 Chimera2-icon64.png

Chimera2-icon32.png:
	convert Chimera2-icon512.png -geometry 32x32 Chimera2-icon32.png

Chimera2-icon16.png:
	convert Chimera2-icon512.png -geometry 16x16 Chimera2-icon16.png
