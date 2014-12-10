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

PROG	= chimera2
STRIP	=
BINMODE	= 755

PY_MAIN = $(PROG)_main.py

GEN_SRCS= frozen.c M___main__.c
OBJS	= $(GEN_SRCS:.c=.o)
INCS	+= $(PYTHON_INCLUDE_DIRS)

LIBS	+= $(PYTHON_LIB)

all: $(PROG)

install: $(PROG)
	$(RSYNC) $(PROG) $(bindir)
	$(RSYNC) $(PY_MAIN) $(PYSITEDIR)
	-if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) $(PROG) $(app_bindir); \
		mkdir $(app_prefix)/MacOS; \
		ln -nfs ../bin/$(PROG) $(app_prefix)/MacOS/build; \
		$(RSYNC) $(PY_MAIN) $(APP_PYSITEDIR); \
	fi
ifeq ($(OS),Darwin)
	-if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) Info.plist $(app_prefix)/Info.plist; \
		mkdir $(app_prefix)/Resources; \
	fi
endif

clean:
	rm -f $(PROG) $(GEN_SRCS) $(OBJS) frozen.c.orig __main__.py
ifeq ($(OS),Darwin)
	rm -f Info.plist
endif

lint:
	$(PYTHON_EXE) -m flake8 *.py

run:
	$(PYTHON_EXE) chimera2_main.py

$(PROG): $(OBJS)
	$(PROG_LINK)

__main__.py: $(PY_MAIN)
	$(PYTHON_EXE) mk__main__.py $<

frozen.c M___main__.c: __main__.py
	$(PYTHON_EXE) mkfrozen.py $(FREEZE_SOURCE) $<
	patch -b -V simple < frozen.patch

ifeq ($(OS),Darwin)
install: Info.plist

Info.plist: mkinfo.py
#	$(bindir)/$(PROG) --nogui mkinfo.py build || rm Info.plist
	$(PYTHON_EXE) mkinfo.py build || rm Info.plist

endif
