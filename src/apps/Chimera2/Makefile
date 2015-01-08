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

GEN_SRCS= frozen.c M___main__.c
OBJS	= $(GEN_SRCS:.c=.o)
INCS	+= $(PYTHON_INCLUDE_DIRS)

LIBS	+= $(PYTHON_LIB)

all: $(PROG)

install: $(PROG)
	$(RSYNC) $(PROG) $(bindir)
	$(RSYNC) $(PY_MAIN) $(PYSITEDIR)
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) $(PROG) $(app_bindir); \
		ln -nfs ../bin/$(PROG) $(app_prefix)/MacOS/$(APP_NAME); \
		$(RSYNC) $(PY_MAIN) $(APP_PYSITEDIR); \
	fi
ifeq ($(OS),Darwin)
	-if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) Info.plist $(app_prefix)/Info.plist; \
	fi
endif

clean:
	rm -f $(PROG) $(GEN_SRCS) $(OBJS) frozen.c.orig __main__.py gensrcs \
		Info.plist

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

ifeq ($(OS),Darwin)
install: Info.plist
endif

Info.plist: mkinfo.py
#	$(bindir)/$(PROG) --nogui mkinfo.py $(APP_NAME) || rm Info.plist
	$(PYTHON_EXE) mkinfo.py $(APP_NAME) || rm Info.plist
