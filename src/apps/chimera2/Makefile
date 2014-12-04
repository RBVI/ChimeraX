# Use this make file to make a frozen verson of chimera for debugging

TOP	= ../../..
include $(TOP)/mk/config.make

# Freeze source was copied from tmp/Python-*/Tools/freeze
FREEZE_SOURCE = $(datadir)/freeze

LDFLAGS	+= -Wl,-rpath,'$$ORIGIN/../lib'

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
	if [ -d "$(app_prefix)" ]; then \
		$(RSYNC) $(PROG) $(app_bindir); \
		$(RSYNC) $(PY_MAIN) $(APP_PYSITEDIR); \
	fi

clean:
	rm -f $(PROG) $(GEN_SRCS) $(OBJS) frozen.c.orig __main__.py

run:
	$(PYTHON_EXE) chimera2_main.py

$(PROG): $(OBJS)
	$(PROG_LINK)

__main__.py: $(PY_MAIN)
	$(PYTHON_EXE) mk__main__.py $<

frozen.c M___main__.c: __main__.py
	$(PYTHON_EXE) mkfrozen.py $(FREEZE_SOURCE) $<
	patch -b < frozen.patch
