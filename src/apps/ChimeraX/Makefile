TOP	= ../../..
include $(TOP)/mk/config.make

ifeq ($(OS),Linux)
LDFLAGS	+= -Wl,-rpath,'$$ORIGIN/../lib'
endif
ifeq ($(OS),Darwin)
LDFLAGS	+= -Wl,-rpath,'@executable_path/../lib'
endif

ifdef WIN32
PROG	= $(APP_NAME).exe
else
PROG	= $(APP_NAME)
endif
STRIP	=
BINMODE	= 755

PY_MAIN = $(APP_NAME)_main.py

ICONS	= ChimeraX-icon.svg ChimeraX-icon512.png
ifeq ($(OS),Linux)
LINUX_ICONS = ChimeraX-icon128.png ChimeraX-icon64.png ChimeraX-icon32.png ChimeraX-icon16.png
ICONS	+= $(LINUX_ICONS)
endif
ifdef WIN32
ICONS	+= $(APP_NAME).ico
endif
SRCS	= launcher.c
OBJS	= $(SRCS:.c=.$(OBJ_EXT))
ifdef WIN32
RCSRCS	= icon.rc
OBJS	+= $(RCSRCS:.rc=.$(OBJ_EXT))
endif
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
ifdef WIN32
ifdef DEBUG
	$(RSYNC) *.pdb $(bindir)
	-if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) *.pdb $(app_bindir); \
	fi
endif
endif
ifeq ($(OS),Darwin)
	# Must use hard link for code signing to work
	-if [ -d "$(app_prefix)" ]; then \
		ln -nf $(app_bindir)/$(PROG) $(app_prefix)/MacOS/$(APP_NAME); \
		$(RSYNC) Info.plist $(app_prefix)/Info.plist; \
		$(RSYNC) chimerax-icon.icns $(app_prefix)/Resources/chimerax-icon.icns; \
	fi
endif

clean:
	rm -f $(PROG) $(OBJS) Info.plist chimerax-icon.icns $(LINUX_ICONS)
ifdef WIN32
	rm -f $(APP_NAME).ico icon.rc $(PROG).manifest $(APP_NAME).pdb vc*.pdb
endif

lint:
	$(PYLINT) *.py

run:
	$(PYTHON_EXE) $(PY_MAIN)

run-nogui:
	$(PYTHON_EXE) $(PY_MAIN) --nogui

$(PROG): $(OBJS)
	$(PROG_LINK)

ChimeraX-icon512.png: ChimeraX-icon.svg
	inkscape --export-png=ChimeraX-icon512.png --export-width=512 --export-background-opacity=0 --without-gui ChimeraX-icon.svg


ifeq ($(OS),Darwin)
install: Info.plist
Info.plist: chimerax-icon.icns
endif

Info.plist: mkinfo.py
#	$(bindir)/$(PROG) --nogui mkinfo.py $(APP_NAME) || rm Info.plist
	$(PYTHON_EXE) mkinfo.py $(APP_NAME) || rm Info.plist

chimerax-icon.icns:	ChimeraX-icon512.png
	mkdir chimerax-icon.iconset
	sips -z 16 16     ChimeraX-icon512.png --out chimerax-icon.iconset/icon_16x16.png
	sips -z 32 32     ChimeraX-icon512.png --out chimerax-icon.iconset/icon_16x16@2x.png
	sips -z 32 32     ChimeraX-icon512.png --out chimerax-icon.iconset/icon_32x32.png
	sips -z 64 64     ChimeraX-icon512.png --out chimerax-icon.iconset/icon_32x32@2x.png
	sips -z 128 128   ChimeraX-icon512.png --out chimerax-icon.iconset/icon_128x128.png
	sips -z 256 256   ChimeraX-icon512.png --out chimerax-icon.iconset/icon_128x128@2x.png
	sips -z 256 256   ChimeraX-icon512.png --out chimerax-icon.iconset/icon_256x256.png
	sips -z 512 512   ChimeraX-icon512.png --out chimerax-icon.iconset/icon_256x256@2x.png
	cp ChimeraX-icon512.png chimerax-icon.iconset/icon_512x512.png
	iconutil -c icns chimerax-icon.iconset
	rm -rf chimerax-icon.iconset

ifeq ($(OS),Linux)
install: $(LINUX_ICONS)
endif

ChimeraX-icon128.png:
	convert ChimeraX-icon512.png -geometry 128x128 ChimeraX-icon128.png

ChimeraX-icon64.png:
	convert ChimeraX-icon512.png -geometry 64x64 ChimeraX-icon64.png

ChimeraX-icon32.png:
	convert ChimeraX-icon512.png -geometry 32x32 ChimeraX-icon32.png

ChimeraX-icon16.png:
	convert ChimeraX-icon512.png -geometry 16x16 ChimeraX-icon16.png

#
# For Windows
#
YEAR = $(shell $(PYTHON_EXE) geticoninfo.py -y)
VER = $(shell $(PYTHON_EXE) geticoninfo.py -v)
PRODVER = $(shell $(PYTHON_EXE) geticoninfo.py -p)
FLAGS = $(shell git show --summary > /dev/null && echo VS_FF_PRERELEASE || echo 0)

$(APP_NAME).ico: $(APP_NAME)-icon512.png
	convert $(APP_NAME)-icon512.png -resize 256x256 \
		-alpha Background -compress zip \
		-define icon:auto-resize=256,128,96,64,48,32,24,16 \
		$(APP_NAME).ico

icon.$(OBJ_EXT): $(APP_NAME).ico

icon.rc: icon.rc.in
	sed -e 's/YEAR/$(YEAR)/' -e 's/CHIMERAX_VERSION/$(VER)/' -e 's/CHIMERAX_PRODUCT/$(PRODVER)/' -e 's/CHIMERAX_FLAGS/$(FLAGS)/' < icon.rc.in > icon.rc
