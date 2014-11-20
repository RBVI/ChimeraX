# Use this make file to make a frozen verson of chimera for debugging

TOP	= ../../..
include $(TOP)/mk/config.make

PY_SOURCE = $(wildcard $(tmpdir)/Python-$(PYTHON_VERSION)*)

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
	cp $(PROG) $(bindir)
	cp $(PY_MAIN) $(PYSITEDIR)

clean:
	rm -f $(PROG) $(GEN_SRCS) $(OBJS) frozen.c.orig __main__.py

run:
	$(bindir)/python3 main.py

$(PROG): $(OBJS)
	$(PROG_LINK)

__main__.py: $(PY_MAIN)
	python3 mk__main__.py $<

frozen.c M___main__.c: __main__.py
	python3 mkfrozen.py $(PY_SOURCE) $<
	patch < frozen.patch
