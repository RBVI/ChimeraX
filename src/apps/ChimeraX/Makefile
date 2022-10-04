TOP	= ../../..
include $(TOP)/mk/config.make

ifeq ($(OS),Linux)
LDFLAGS	+= -Wl,-rpath,'$$ORIGIN/../lib'
endif
ifeq ($(OS),Darwin)
LDFLAGS	+= -Wl,-rpath,'@executable_path/../lib'
endif

ifdef WIN32
ifeq ($(DESKTOP),1)
PROG	= $(APP_NAME).exe
else
PROG	= $(APP_NAME)-console.exe
endif
else
PROG	= $(APP_NAME)
endif
STRIP	=
BINMODE	= 755

ICONS	= ChimeraX-icon.svg ChimeraX-icon512.png
ifeq ($(OS),Linux)
LINUX_ICONS = ChimeraX-icon128.png ChimeraX-icon64.png ChimeraX-icon32.png ChimeraX-icon16.png
ICONS	+= $(LINUX_ICONS)
endif
ifdef WIN32
ICONS	+= $(APP_NAME).ico
endif
SHARED_SRCS	= launcher.c
RCSRCS	= icon.rc
ifdef WIN32
SHARED_OBJS	= $(SHARED_SRCS:.c=.obj) $(RCSRCS:.rc=.$(OBJ_EXT))
ifeq ($(DESKTOP),1)
SRCS	= $(SHARED_SRCS) main_desktop.c
else
SRCS	= $(SHARED_SRCS) main_console.c
endif
OBJS	= $(SRCS:.c=.$(OBJ_EXT)) $(RCSRCS:.rc=.$(OBJ_EXT))
else
SRCS	= $(SHARED_SRCS) main_console.c
OBJS	= $(SRCS:.c=.$(OBJ_EXT))
endif
INCS	+= $(PYTHON_INCLUDE_DIRS)
LIBS	+= $(PYTHON_LIB)

all: $(PROG)

ifdef WIN32
install:	install-desktop install-console install-others

install-desktop:	$(SHARED_OBJS)
	$(MAKE) real-install DESKTOP=1

install-console:	$(SHARED_OBJS)
	$(MAKE) real-install DESKTOP=0

real-install: $(PROG)
	$(RSYNC) $(PROG) $(bindir)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) $(PROG) $(app_bindir); \
	fi
ifdef DEBUG
	$(RSYNC) *.pdb $(bindir)
	-if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) *.pdb $(app_bindir); \
	fi
endif

install-others: $(ICONS)
	$(RSYNC) $(ICONS) $(datadir)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) $(ICONS) $(app_datadir); \
	fi
else
install: $(PROG) $(ICONS) $(APP_NAME).1
	$(RSYNC) $(PROG) $(bindir)
	$(RSYNC) $(ICONS) $(datadir)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) $(PROG) $(app_bindir); \
		$(RSYNC) $(ICONS) $(app_datadir); \
	fi
	mkdir -p $(datadir)/man/man1
	$(RSYNC) $(APP_NAME).1 $(datadir)/man/man1
	-if [ -d "$(app_prefix)" ]; then \
		mkdir -p $(app_datadir)/man/man1; \
		$(RSYNC) $(APP_NAME).1 $(app_datadir)/man/man1; \
	fi
ifeq ($(OS),Darwin)
	# Must use hard link for code signing to work
	-if [ -d "$(app_prefix)" ]; then \
		ln -nf $(app_bindir)/$(PROG) $(app_prefix)/MacOS/$(APP_NAME); \
	fi

install-plist: mkinfo.py chimerax-icon.icns
	# use Chimera executable in this pass so that we pick up registered formats (and 'session' is available)
	$(wildcard $(TOP)/ChimeraX*.app)/Contents/bin/ChimeraX --nogui --script "mkinfo.py $(APP_NAME)" --exit || (rm Info.plist ; exit 1)
	$(RSYNC) Info.plist $(app_prefix)/Info.plist
	$(RSYNC) chimerax-icon.icns $(app_prefix)/Resources/chimerax-icon.icns
endif
endif

clean:
	rm -f $(PROG) $(OBJS) Info.plist chimerax-icon.icns $(LINUX_ICONS)
ifdef WIN32
	rm -f $(APP_NAME).ico icon.rc $(PROG).manifest $(APP_NAME).pdb vc*.pdb \
		$(PROG:.exe=.exp) $(PROG:.exe=.lib) main_desktop.$(OBJ_EXT)
ifneq ($(DESKTOP),1)
	$(MAKE) clean DESKTOP=1
endif
endif

pylint:
	$(PYLINT) *.py

# 64-bit Linux dies in exception handling if we don't link with the C++ compiler
$(PROG): $(OBJS)
	$(PROG_LINK)
ifeq ($(OS),Linux)
	# convert to RPATH
	if objdump -p $(PROG) | grep RUNPATH > /dev/null; then \
		$(bindir)/chrpath -C $(PROG); \
	fi
endif
ifeq ($(OS),Darwin)
	# Add RPATH so Python ssl module can find libssl.dylib.
	# It needs Python.Framework in the RPATH
	install_name_tool -add_rpath '@executable_path/../Library/Frameworks/Python.framework' $(PROG)
endif

ChimeraX-icon512.png: ChimeraX-icon.svg
	inkscape --export-png=ChimeraX-icon512.png --export-width=512 --export-background-opacity=0 --without-gui ChimeraX-icon.svg

chimerax-icon.icns:	ChimeraX-icon512.png
	-mkdir chimerax-icon.iconset
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

ifdef WIN32

YEAR = $(shell $(APP_PYTHON_EXE) geticoninfo.py -y)
VER = $(shell $(APP_PYTHON_EXE) geticoninfo.py -v)
PRODVER = $(shell $(APP_PYTHON_EXE) geticoninfo.py -p)
RELEASE =  $(shell $(APP_PYTHON_EXE) geticoninfo.py -r)
ifeq ($(RELEASE),prerelease)
FLAGS = VS_FF_PRERELEASE
else
FLAGS = 0
endif

$(APP_NAME).ico: $(APP_NAME)-icon512.png
	convert $(APP_NAME)-icon512.png -resize 256x256 \
		-alpha Background -compress zip \
		-define icon:auto-resize=256,128,96,64,48,32,24,16 \
		$(APP_NAME).ico

icon.$(OBJ_EXT): $(APP_NAME).ico

icon.rc: icon.rc.in
	sed -e 's/YEAR/$(YEAR)/' -e 's/CHIMERAX_VERSION/$(VER)/' -e 's/CHIMERAX_PRODUCT/$(PRODVER)/' -e 's/CHIMERAX_FLAGS/$(FLAGS)/' < icon.rc.in > icon.rc

endif
